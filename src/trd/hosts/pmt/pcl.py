import logging
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from . import utils as loss_utils
from .utils import _distributed_all_gather, safe_logits


logger = logging.getLogger('PMT')


class _SampledInfoNCEStopGradKeys(torch.autograd.Function):
    """
    Sampled InfoNCE chunk loss with stop-grad keys.

    Forward computes the same logits/loss as the regular path:
      L = mean(logsumexp([pos, negs]) - pos)

    Backward returns gradients only for anchors while storing compact backward
    statistics ([C, D]) instead of all sampled negatives ([C, n_neg, D]).
    """

    @staticmethod
    def forward(
        ctx,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        tau: torch.Tensor | float,
        debug: bool,
        use_probs_no_cat: bool,
    ) -> torch.Tensor:
        # Keep math in fp32 for stability and parity with safe_logits.
        anchor_f = anchors.float()
        pos_f = positives.float()
        neg_f = negatives.float()

        if isinstance(tau, torch.Tensor):
            tau_t = tau if tau.dtype == torch.float32 else tau.float()
            if tau_t.device != anchors.device:
                tau_t = tau_t.to(device=anchors.device)
        else:
            tau_t = torch.tensor(float(tau), device=anchors.device, dtype=torch.float32)

        # Raw similarities before temperature scaling/clamp.
        pos_raw = torch.sum(anchor_f * pos_f, dim=1)
        neg_raw = torch.bmm(neg_f, anchor_f.unsqueeze(-1)).squeeze(-1)

        # Match regular logits path exactly.
        pos_logits = safe_logits(pos_raw.unsqueeze(1), tau_t, debug=bool(debug)).squeeze(1)
        neg_logits = safe_logits(neg_raw, tau_t, debug=bool(debug))

        neg_logsumexp = torch.logsumexp(neg_logits, dim=1)
        log_denominator = torch.logaddexp(pos_logits, neg_logsumexp)
        loss = (log_denominator - pos_logits).mean()

        # Compact backward stats:
        # dL/da = (1/tau) * (sum_j mask_j * p_j * k_j - mask_pos * k_pos), averaged over C.
        clamp = float(loss_utils.LOGIT_CLAMP)
        scaled_pos = pos_raw / tau_t
        scaled_neg = neg_raw / tau_t
        pos_mask = ((scaled_pos >= -clamp) & (scaled_pos <= clamp)).to(dtype=anchor_f.dtype)
        neg_mask = ((scaled_neg >= -clamp) & (scaled_neg <= clamp)).to(dtype=anchor_f.dtype)

        if bool(use_probs_no_cat):
            # Candidate micro-optimization: compute probabilities directly from the
            # log-normalizer and avoid cat+softmax over [C, 1+n_neg].
            log_z = log_denominator
            pos_prob = torch.exp(pos_logits - log_z)
            neg_prob = torch.exp(neg_logits - log_z.unsqueeze(1))
        else:
            probs = torch.softmax(torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1), dim=1)
            pos_prob = probs[:, 0]
            neg_prob = probs[:, 1:]

        pos_coeff = pos_mask * pos_prob
        neg_coeff = neg_mask * neg_prob

        expected_keys = pos_f * pos_coeff.unsqueeze(1)
        expected_keys = expected_keys + torch.bmm(neg_coeff.unsqueeze(1), neg_f).squeeze(1)
        masked_pos = pos_f * pos_mask.unsqueeze(1)

        ctx.save_for_backward(expected_keys, masked_pos, tau_t)
        ctx.anchor_dtype = anchors.dtype
        ctx.chunk_size = int(anchors.size(0))
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        expected_keys, masked_pos, tau_t = ctx.saved_tensors
        chunk_size = max(int(ctx.chunk_size), 1)

        grad_anchor = (expected_keys - masked_pos) / tau_t
        grad_anchor = grad_anchor / float(chunk_size)
        grad_anchor = grad_anchor * grad_output.to(dtype=grad_anchor.dtype)
        grad_anchor = grad_anchor.to(dtype=ctx.anchor_dtype)

        return grad_anchor, None, None, None, None, None


class PMAContrastiveLoss(nn.Module):
    """
    A chunk-based InfoNCE loss over all PMA states from two views,
    with optional exclusion of local states from the negative pool.

    If exclude_local_sample=True (default):
      - all states belonging to the same original sequence as any anchor
        are removed from the negative pool (current behaviour).

    If exclude_local_sample=False:
      - only the exact self/positive rows are removed; other states from
        the same sequence may appear as negatives (harder objective).

    This is analogous to patch/window-based negative sampling, but
    specifically for PMA states shaped [B, T, n_states, D].

    Configuration is typically drawn from cfg.pma_contrastive_loss.*:
      - use_per_state_loss: bool
      - temperature: float
      - chunk_size: int
      - max_chunk_halves: int
      - n_negative_samples: int
      - weight: float (if you want to scale the final loss in your pipeline)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        end_temperature: float = 0.05,
        temp_schedule_mode: str = "absolute",
        chunk_size: int = 512,
        max_chunk_halves: int = 3,
        n_negative_samples: int = 64,
        negative_sampling_mode: str = "random",
        negatives_per_source: int = 2,
        exclude_local_sample: bool = True,
        restrict_negatives_to_state_slot: bool = False,
        enable_small_batch_combined_pool: bool = False,
        sampling_seed: int | None = None,
        use_distributed_row_weighting: bool = True,
        use_compact_backward_stats: bool = False,
        use_compact_probs_no_cat: bool = False,
        use_streamed_full_denom: bool = False,
        full_denom_key_tile_size: int = 4096,
        debug: bool = False,
    ):
        """
        Contrastive loss for PMA state representations.

        Args:
            temperature: Starting temperature for softmax
            end_temperature: Ending temperature after scheduling
            chunk_size: Initial chunk size for batched processing
            max_chunk_halves: Maximum times to halve chunk_size if OOM occurs
            n_negative_samples: Number of negative examples per anchor
            negative_sampling_mode: Negative sampler to use: "random" or "source_stratified".
            negatives_per_source: Maximum sampled rows per non-anchor source for source_stratified mode.
            exclude_local_sample: If True, exclude all states from the same original sequence (batch item).
                                  If False, only the exact self/positive rows are excluded.
            restrict_negatives_to_state_slot: If True, each state slot s only competes against
                                              pooled rows from slot s (no cross-slot negatives).
            enable_small_batch_combined_pool: If True, allow a non-DDP final-state-only small-batch
                                              fallback that combines both views into the negative pool.
                                              Default False to keep single-GPU and DDP objectives aligned.
            sampling_seed: Optional base seed for negative sampling RNG. When provided,
                           rank is added automatically so DDP ranks do not share identical streams.
            use_distributed_row_weighting: If True, apply per-rank row-count weighting in distributed
                                           mode so uneven local batch sizes still match global-mean
                                           gradient semantics.
            use_compact_backward_stats: If True, use a custom autograd path that preserves
                                        sampled InfoNCE semantics while storing compact
                                        [C, D] backward statistics instead of full negatives.
            use_compact_probs_no_cat: If True, use the candidate compact-forward
                                      probability path (exp(logit-log_z)) instead of
                                      softmax(cat([pos, neg])) inside custom autograd.
                                      Default False to keep the established Phase-2.1 path.
            use_streamed_full_denom: If True, compute exact InfoNCE denominators by
                                     streaming over key tiles instead of sampling negatives.
            full_denom_key_tile_size: Number of keys per streamed tile when
                                      use_streamed_full_denom=True.
            debug: Enable debug logging
        """
        super().__init__()  # Initialize the parent nn.Module class

        self.temperature = temperature
        if temp_schedule_mode == "fraction":
            self.end_temperature = temperature * end_temperature
        else:
            self.end_temperature = end_temperature
        self._current_epoch = 0
        self._sched_len = 0  # Will be set via set_epoch

        self.chunk_size = chunk_size
        self.max_chunk_halves = max_chunk_halves
        self.n_negative_samples = n_negative_samples
        self.negative_sampling_mode = self._normalize_negative_sampling_mode(negative_sampling_mode)
        self.negatives_per_source = int(negatives_per_source)
        if self.negatives_per_source <= 0:
            raise ValueError(f"negatives_per_source must be > 0, got {self.negatives_per_source}")
        self.exclude_local_sample = exclude_local_sample
        self.restrict_negatives_to_state_slot = bool(restrict_negatives_to_state_slot)
        self.enable_small_batch_combined_pool = bool(enable_small_batch_combined_pool)
        self.use_distributed_row_weighting = bool(use_distributed_row_weighting)
        self.use_compact_backward_stats = bool(use_compact_backward_stats)
        self.use_compact_probs_no_cat = bool(use_compact_probs_no_cat)
        self.use_streamed_full_denom = bool(use_streamed_full_denom)
        self.full_denom_key_tile_size = int(full_denom_key_tile_size)
        if self.full_denom_key_tile_size <= 0:
            raise ValueError(f"full_denom_key_tile_size must be > 0, got {self.full_denom_key_tile_size}")
        self.debug = debug
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger('PMT')

        # Check distributed
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0

        # Reusable random generator - create with device matching our tensors
        # We'll initialize this in compute_loss when we know the actual device
        self._g = None
        self._g_device = None
        base_seed = random.randint(0, 2**31 - 1) if sampling_seed is None else int(sampling_seed)
        self._seed = int(base_seed) + int(self.rank)
        self._seed_rank = int(self.rank)

        # Cache distributed geometry checks; in non-debug runs we only re-check when local
        # geometry changes to keep the hot path free of extra collectives.
        self._last_validated_local_geometry: tuple[int, int, int, int] | None = None

        # Runtime chunk telemetry for benchmark scripts.
        self._last_chunk_size_before = int(chunk_size) if chunk_size else 0
        self._last_effective_chunk_size = int(chunk_size) if chunk_size else 0
        self._last_chunk_halves = 0

        if self.debug and self.is_distributed:
            self.logger.debug("PMAContrastiveLoss detected distributed training.")

        # Register temperature as a buffer for torch.compile compatibility
        self.register_buffer("_tau", torch.tensor(float(temperature)))

        if self.debug:
            self.logger.debug(
                f"Initialized PMAContrastiveLoss with: "
                f"temp={self.temperature}, chunk_size={self.chunk_size}, "
                f"n_neg={self.n_negative_samples}, exclude_local={self.exclude_local_sample}, "
                f"negative_sampling_mode={self.negative_sampling_mode}, "
                f"negatives_per_source={self.negatives_per_source}, "
                f"restrict_slot={self.restrict_negatives_to_state_slot}, "
                f"small_batch_combined_pool={self.enable_small_batch_combined_pool}, "
                f"row_weighting={self.use_distributed_row_weighting}, "
                f"compact_backward={self.use_compact_backward_stats}, "
                f"compact_probs_no_cat={self.use_compact_probs_no_cat}, "
                f"streamed_full_denom={self.use_streamed_full_denom}, "
                f"full_denom_key_tile_size={self.full_denom_key_tile_size}"
            )
            if self.use_streamed_full_denom and self.use_compact_backward_stats:
                self.logger.debug(
                    "PMAContrastiveLoss streamed full-denominator path is active; "
                    "use_compact_backward_stats has no effect in this mode."
                )

    @staticmethod
    def _normalize_negative_sampling_mode(mode: str) -> str:
        """
        Normalize and validate the negative sampler mode.
        """
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {"uniform": "random", "legacy": "random"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"random", "source_stratified"}:
            raise ValueError(f"negative_sampling_mode must be one of {{'random', 'source_stratified'}}, got {mode!r}")
        return normalized

    def set_sampling_seed(self, seed: int) -> None:
        """
        Override sampling seed at runtime (rank offset is applied automatically).
        """
        self._seed = int(seed) + int(self.rank)
        if self._g is not None:
            self._g.manual_seed(self._seed)

    def set_epoch(self, epoch: int, schedule_epochs: int = 0):
        """
        Update the temperature based on the current epoch.

        Args:
            epoch: Current training epoch
            schedule_epochs: Total number of epochs for the schedule (if 0, uses the value from initialization)
        """
        self._current_epoch = epoch

        # Set schedule length if provided
        if schedule_epochs > 0:
            self._sched_len = schedule_epochs

        # Early-exit when temperature scheduling is disabled
        if self._sched_len == 0:
            return

        # Update temperature based on schedule
        if self._sched_len > 0:
            t = min(epoch / self._sched_len, 1.0)
            new_temp = self.temperature + t * (self.end_temperature - self.temperature)
            with torch.no_grad():
                self._tau.fill_(new_temp)

    def compute_loss(self, states1: torch.Tensor, states2: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between PMA states from two views.

        Args:
            states1: Shape [B, T, n_states, D] - PMA states from first view
            states2: Shape [B, T, n_states, D] - PMA states from second view

        Returns:
            Scalar contrastive loss
        """
        if states1.shape != states2.shape:
            raise ValueError(f"PMAContrastiveLoss expects matching shapes, got {states1.shape} vs {states2.shape}")

        # Re-check distributed state at call time in case initialization changed after module construction.
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        if self.rank != self._seed_rank:
            self._seed = int(self._seed) + int(self.rank) - int(self._seed_rank)
            self._seed_rank = int(self.rank)
            # Force generator re-init with the rank-adjusted seed.
            self._g = None
            self._g_device = None

        device = states1.device
        self.device = device  # Keep self.device in sync when module is moved to a different device.
        # Keep temperature buffer on the active device; avoid per-forward temporary copies.
        if self._tau.device != device:
            self._tau = self._tau.to(device=device)
        tau = self._tau

        self._ensure_generator(device)

        B, T, S, D = states1.shape

        # Each sample contributes T*S rows to the flattened tensor.
        rows_per_sample = T * S

        # "Final state only" is the case where temporal dimension is collapsed.
        is_final_state_only = T == 1

        if is_final_state_only and self.debug:
            self.logger.debug(f"Computing state contrastive loss using only final PMA state: shape={states1.shape}")

        N = B * T * S  # Total number of states per view.

        # Flatten.
        z1_local = states1.reshape(N, D)
        z2_local = states2.reshape(N, D)

        self._maybe_validate_distributed_geometry(
            z2_local=z2_local, rows_per_sample=rows_per_sample, batch_size=B, timesteps=T, state_slots=S
        )

        # Sanity check to catch corrupted shapes early.
        if z2_local.numel() > 1e8 or z1_local.numel() > 1e8:
            raise RuntimeError(
                f"PMAContrastiveLoss received unrealistically large tensor (z1_local={tuple(z1_local.shape)}, z2_local={tuple(z2_local.shape)})"
            )

        # Normalize before similarity computations.
        z1_local = F.normalize(z1_local, p=2, dim=-1)
        z2_local = F.normalize(z2_local, p=2, dim=-1)

        # Use symmetric stop-grad directions so single-GPU and DDP follow the same objective.
        loss_fwd = self._compute_directional_loss(
            anchors_local=z1_local,
            keys_local=z2_local,
            rows_per_sample=rows_per_sample,
            tau=tau,
            is_final_state_only=is_final_state_only,
            batch_size=B,
            num_state_slots=S,
        )
        loss_rev = self._compute_directional_loss(
            anchors_local=z2_local,
            keys_local=z1_local,
            rows_per_sample=rows_per_sample,
            tau=tau,
            is_final_state_only=is_final_state_only,
            batch_size=B,
            num_state_slots=S,
        )
        loss = 0.5 * (loss_fwd + loss_rev)

        if self.is_distributed and self.use_distributed_row_weighting:
            loss = self._apply_distributed_row_weighting(loss=loss, local_rows=int(z1_local.size(0)), device=device)

        return loss

    def _maybe_validate_distributed_geometry(
        self, *, z2_local: torch.Tensor, rows_per_sample: int, batch_size: int, timesteps: int, state_slots: int
    ) -> None:
        """
        Validate DDP geometry assumptions with collective-safe gating.

        In non-debug mode we only run the heavy all-gather validation when any rank's
        local geometry has changed. The "any rank needs validation" decision itself is
        synchronized via all-reduce so ranks cannot diverge on collective participation.
        """
        if not self.is_distributed:
            return

        local_geometry = (int(z2_local.shape[1]), int(rows_per_sample), int(timesteps), int(state_slots))
        need_validation_local = float(self.debug or (self._last_validated_local_geometry != local_geometry))
        gate_tensor = torch.tensor([need_validation_local], device=z2_local.device, dtype=torch.float32)
        if hasattr(dist, "all_reduce"):
            dist.all_reduce(gate_tensor)
        if float(gate_tensor.detach().cpu().item()) <= 0.0:
            return

        device = z2_local.device
        info_local = torch.tensor(
            [
                int(z2_local.shape[0]),
                int(z2_local.shape[1]),
                int(z2_local.numel()),
                int(rows_per_sample),
                int(timesteps),
                int(state_slots),
                int(batch_size),
            ],
            device=device,
            dtype=torch.long,
        )
        info_world = [torch.zeros_like(info_local) for _ in range(dist.get_world_size())]
        dist.all_gather(info_world, info_local)
        info_table = torch.stack(info_world, dim=0).cpu().tolist()
        info_rows = [tuple(int(v) for v in row) for row in info_table]

        max_numel = max(x[2] for x in info_rows)
        min_numel = min(x[2] for x in info_rows)
        if min_numel < 0 or max_numel > 1_000_000_000:
            raise RuntimeError(
                f"PMAContrastiveLoss detected invalid tensor sizes across ranks: {info_rows} "
                f"(rank {self.rank}, local z2_local shape={tuple(z2_local.shape)})"
            )
        if any((x[0] * x[1]) != x[2] for x in info_rows):
            raise RuntimeError(f"PMAContrastiveLoss received inconsistent distributed tensor metadata: {info_rows}")

        dims = {x[1] for x in info_rows}
        if len(dims) != 1:
            rank_dump = ", ".join(
                f"r{i}(N={x[0]},D={x[1]},B={x[6]},T={x[4]},S={x[5]},rows_per_sample={x[3]})"
                for i, x in enumerate(info_rows)
            )
            raise RuntimeError(
                "PMAContrastiveLoss requires a consistent embedding dimension across ranks; "
                f"observed distributed geometry: {rank_dump}"
            )

        ts_pairs = {(x[4], x[5]) for x in info_rows}
        if len(ts_pairs) != 1:
            rank_dump = ", ".join(
                f"r{i}(B={x[6]},T={x[4]},S={x[5]},N={x[0]},rows_per_sample={x[3]})" for i, x in enumerate(info_rows)
            )
            raise RuntimeError(
                "PMAContrastiveLoss requires consistent T and S across ranks for row-aligned PMA objectives; "
                f"observed distributed geometry: {rank_dump}"
            )

        if self.exclude_local_sample:
            rows_per_sample_world = {x[3] for x in info_rows}
            if len(rows_per_sample_world) != 1:
                rank_dump = ", ".join(
                    f"r{i}(rows_per_sample={x[3]},B={x[6]},T={x[4]},S={x[5]},N={x[0]})" for i, x in enumerate(info_rows)
                )
                raise RuntimeError(
                    "PMAContrastiveLoss with exclude_local_sample=True requires identical rows_per_sample "
                    f"across ranks; observed distributed geometry: {rank_dump}"
                )

        self._last_validated_local_geometry = local_geometry

    def _apply_distributed_row_weighting(
        self, *, loss: torch.Tensor, local_rows: int, device: torch.device
    ) -> torch.Tensor:
        """
        Scale local loss so DDP's per-rank gradient averaging matches a true global-row mean
        even when local row counts differ across ranks.
        """
        if not self.is_distributed or not hasattr(dist, "all_reduce"):
            return loss

        local_rows_tensor = torch.tensor([max(int(local_rows), 0)], device=device, dtype=torch.float32)
        global_rows_tensor = local_rows_tensor.clone()
        dist.all_reduce(global_rows_tensor)

        world_size = float(dist.get_world_size())
        scale = (local_rows_tensor * world_size) / global_rows_tensor.clamp_min(1.0)
        return loss * scale.to(dtype=loss.dtype).squeeze(0)

    def _ensure_generator(self, device: torch.device) -> None:
        """
        Keep a per-device RNG so module/device moves do not leave a stale generator.
        """
        if self._g is not None and self._g_device == device:
            return
        try:
            self._g = torch.Generator(device=device)
        except (TypeError, RuntimeError):
            self._g = torch.Generator()
        self._g.manual_seed(self._seed)
        self._g_device = device

    def _compute_directional_loss(
        self,
        anchors_local: torch.Tensor,
        keys_local: torch.Tensor,
        rows_per_sample: int,
        tau: torch.Tensor | float,
        is_final_state_only: bool,
        batch_size: int,
        num_state_slots: int,
    ) -> torch.Tensor:
        """
        Compute one InfoNCE direction with stop-grad keys.
        """
        use_combined_pool = (
            self.enable_small_batch_combined_pool
            and (not self.is_distributed)
            and is_final_state_only
            and batch_size <= 4
        )
        local_rows = anchors_local.size(0)
        if local_rows == 0:
            return anchors_local.sum() * 0.0
        device = anchors_local.device

        if use_combined_pool:
            if self.debug:
                self.logger.debug("Using local 2-view combined pool for small final-state batch.")
            pool = torch.cat([anchors_local, keys_local], dim=0).detach()
            positive_indices = torch.arange(local_rows, device=device, dtype=torch.long) + local_rows
            self_indices = torch.arange(local_rows, device=device, dtype=torch.long)
            views = 2
        else:
            if self.is_distributed:
                pool, local_offset, _ = _distributed_all_gather(keys_local)
            else:
                pool, local_offset = keys_local, 0
            pool = pool.detach()
            positive_indices = torch.arange(local_rows, device=device, dtype=torch.long) + int(local_offset)
            self_indices = None
            views = 1

        if pool.numel() == 0:
            return anchors_local.sum() * 0.0

        if self.debug and positive_indices.numel() > 0:
            if bool((positive_indices >= pool.size(0)).any().item()):
                max_pos = int(positive_indices.max().detach().cpu().item())
                raise RuntimeError(
                    f"PMAContrastiveLoss positive index out of range: max={max_pos} " f"pool_size={pool.size(0)}"
                )

        if self.restrict_negatives_to_state_slot and int(num_state_slots) > 1:
            return self._compute_state_slot_directional_loss(
                anchors_local=anchors_local,
                pool=pool,
                positive_indices=positive_indices,
                self_indices=self_indices,
                rows_per_sample=rows_per_sample,
                views=views,
                tau=tau,
                num_state_slots=int(num_state_slots),
            )

        return self._compute_chunked_directional_loss(
            anchors_local=anchors_local,
            pool=pool,
            positive_indices=positive_indices,
            self_indices=self_indices,
            rows_per_sample=rows_per_sample,
            views=views,
            tau=tau,
        )

    def _compute_state_slot_directional_loss(
        self,
        anchors_local: torch.Tensor,
        pool: torch.Tensor,
        positive_indices: torch.Tensor,
        self_indices: torch.Tensor | None,
        rows_per_sample: int,
        views: int,
        tau: torch.Tensor | float,
        num_state_slots: int,
    ) -> torch.Tensor:
        """
        Compute one directional loss where each state slot s only competes against
        rows from state slot s in the pool (no cross-slot negatives).
        """
        S = int(num_state_slots)
        if S <= 1:
            return self._compute_chunked_directional_loss(
                anchors_local=anchors_local,
                pool=pool,
                positive_indices=positive_indices,
                self_indices=self_indices,
                rows_per_sample=rows_per_sample,
                views=views,
                tau=tau,
            )

        if anchors_local.size(0) % S != 0:
            raise RuntimeError(
                f"restrict_negatives_to_state_slot=True requires anchors rows divisible by S. "
                f"anchors_rows={anchors_local.size(0)} S={S}"
            )
        if pool.size(0) % S != 0:
            raise RuntimeError(
                f"restrict_negatives_to_state_slot=True requires pool rows divisible by S. "
                f"pool_rows={pool.size(0)} S={S}"
            )
        if rows_per_sample % S != 0:
            raise RuntimeError(
                f"restrict_negatives_to_state_slot=True requires rows_per_sample divisible by S. "
                f"rows_per_sample={rows_per_sample} S={S}"
            )

        slot_rows_per_sample = rows_per_sample // S
        slot_losses: list[torch.Tensor] = []
        slot_weights: list[int] = []

        for slot in range(S):
            anchors_slot = anchors_local[slot::S]
            if anchors_slot.numel() == 0:
                continue
            pool_slot = pool[slot::S]

            pos_full_slot = positive_indices[slot::S]
            if (pos_full_slot.remainder(S) != slot).any():
                raise RuntimeError("PMAContrastiveLoss positive index slot alignment invariant violated.")
            positive_slot = torch.div(pos_full_slot - slot, S, rounding_mode="floor")

            self_slot = None
            if self_indices is not None:
                self_full_slot = self_indices[slot::S]
                if (self_full_slot.remainder(S) != slot).any():
                    raise RuntimeError("PMAContrastiveLoss self index slot alignment invariant violated.")
                self_slot = torch.div(self_full_slot - slot, S, rounding_mode="floor")

            if self.debug and positive_slot.numel() > 0:
                if bool((positive_slot >= int(pool_slot.size(0))).any().item()):
                    max_pos = int(positive_slot.max().detach().cpu().item())
                    raise RuntimeError(
                        f"PMAContrastiveLoss slot positive index out of range: slot={slot}, "
                        f"max={max_pos}, pool_slot={int(pool_slot.size(0))}"
                    )

            slot_loss = self._compute_chunked_directional_loss(
                anchors_local=anchors_slot,
                pool=pool_slot,
                positive_indices=positive_slot,
                self_indices=self_slot,
                rows_per_sample=slot_rows_per_sample,
                views=views,
                tau=tau,
            )
            slot_losses.append(slot_loss)
            slot_weights.append(int(anchors_slot.size(0)))

        if not slot_losses:
            return anchors_local.sum() * 0.0

        weights = torch.tensor(slot_weights, device=anchors_local.device, dtype=torch.float32)
        losses = torch.stack(slot_losses)
        return (losses * weights).sum() / weights.sum().clamp_min(1.0)

    def _compute_chunked_directional_loss(
        self,
        anchors_local: torch.Tensor,
        pool: torch.Tensor,
        positive_indices: torch.Tensor,
        self_indices: torch.Tensor | None,
        rows_per_sample: int,
        views: int,
        tau: torch.Tensor | float,
    ) -> torch.Tensor:
        """
        Process anchors in chunks and compute a weighted mean InfoNCE loss.
        """
        N = anchors_local.size(0)
        initial_chunk_size = max(1, int(self.chunk_size)) if self.chunk_size else max(1, N)
        chunk_size = initial_chunk_size
        current_chunk_halves = 0
        self._last_chunk_size_before = int(initial_chunk_size)
        self._last_effective_chunk_size = int(initial_chunk_size)
        self._last_chunk_halves = 0

        if self.use_streamed_full_denom:
            return self._compute_chunked_streamed_full_denom_directional_loss(
                anchors_local=anchors_local,
                pool=pool,
                positive_indices=positive_indices,
                self_indices=self_indices,
                rows_per_sample=rows_per_sample,
                views=views,
                tau=tau,
                initial_chunk_size=initial_chunk_size,
            )

        # Pre-sample negatives once per directional pass so chunk partition changes
        # (including OOM backoff) do not alter sampled negative identities.
        all_neg_indices = self._sample_negatives_for_chunk(
            positive_indices=positive_indices,
            self_indices=self_indices,
            global_size=pool.size(0),
            rows_per_sample=rows_per_sample,
            views=views,
        )
        if all_neg_indices is None:
            return anchors_local.sum() * 0.0

        while True:
            oom_happened = False
            loss_chunks: list[torch.Tensor] = []
            weight_chunks: list[int] = []

            try:
                for start_idx in range(0, N, chunk_size):
                    end_idx = min(start_idx + chunk_size, N)
                    c_size = end_idx - start_idx
                    if c_size <= 0:
                        continue

                    anchor_chunk = anchors_local[start_idx:end_idx]
                    pos_chunk = positive_indices[start_idx:end_idx]
                    neg_indices = all_neg_indices[start_idx:end_idx]
                    neg_count = int(neg_indices.size(1))
                    if neg_count <= 0:
                        continue

                    positives = pool.index_select(0, pos_chunk)
                    flat_neg = neg_indices.reshape(-1)
                    negatives = pool.index_select(0, flat_neg).view(c_size, neg_count, -1)

                    if self.use_compact_backward_stats:
                        loss_chunks.append(
                            _SampledInfoNCEStopGradKeys.apply(
                                anchor_chunk,
                                positives,
                                negatives,
                                tau,
                                bool(self.debug),
                                bool(self.use_compact_probs_no_cat),
                            )
                        )
                    else:
                        pos_sims = torch.sum(anchor_chunk * positives, dim=1)
                        neg_sims = torch.bmm(negatives, anchor_chunk.unsqueeze(-1)).squeeze(-1)

                        pos_sims = safe_logits(pos_sims.unsqueeze(1), tau, debug=self.debug).squeeze(1)
                        neg_sims = safe_logits(neg_sims, tau, debug=self.debug)

                        neg_logsumexp = torch.logsumexp(neg_sims, dim=1)
                        log_denominator = torch.logaddexp(pos_sims, neg_logsumexp)
                        loss_chunks.append((log_denominator - pos_sims).mean())
                    weight_chunks.append(c_size)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and current_chunk_halves < self.max_chunk_halves:
                    oom_happened = True
                else:
                    raise

            if not oom_happened:
                break

            current_chunk_halves += 1
            chunk_size = max(1, chunk_size // 2)
            if self.debug:
                self.logger.debug(f"OOM occurred. Reducing chunk size to {chunk_size}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._last_effective_chunk_size = int(chunk_size)
        self._last_chunk_halves = int(current_chunk_halves)

        if not loss_chunks:
            return anchors_local.sum() * 0.0

        weighted_sum = torch.sum(torch.stack([w * lc for w, lc in zip(weight_chunks, loss_chunks)]))
        return weighted_sum / sum(weight_chunks)

    def _compute_chunked_streamed_full_denom_directional_loss(
        self,
        anchors_local: torch.Tensor,
        pool: torch.Tensor,
        positive_indices: torch.Tensor,
        self_indices: torch.Tensor | None,
        rows_per_sample: int,
        views: int,
        tau: torch.Tensor | float,
        initial_chunk_size: int,
    ) -> torch.Tensor:
        """
        Exact stop-grad InfoNCE with streamed denominator over key tiles.
        """
        N = anchors_local.size(0)
        chunk_size = initial_chunk_size
        current_chunk_halves = 0

        while True:
            oom_happened = False
            loss_chunks: list[torch.Tensor] = []
            weight_chunks: list[int] = []

            try:
                for start_idx in range(0, N, chunk_size):
                    end_idx = min(start_idx + chunk_size, N)
                    c_size = end_idx - start_idx
                    if c_size <= 0:
                        continue

                    anchor_chunk = anchors_local[start_idx:end_idx]
                    pos_chunk = positive_indices[start_idx:end_idx]
                    self_chunk = self_indices[start_idx:end_idx] if self_indices is not None else None

                    positives = pool.index_select(0, pos_chunk)
                    pos_sims = safe_logits(torch.sum(anchor_chunk * positives, dim=1), tau, debug=self.debug)
                    log_denominator = self._stream_full_denom_log_z(
                        anchors=anchor_chunk,
                        pool=pool,
                        positive_indices=pos_chunk,
                        self_indices=self_chunk,
                        rows_per_sample=rows_per_sample,
                        views=views,
                        tau=tau,
                    )
                    loss_chunks.append((log_denominator - pos_sims).mean())
                    weight_chunks.append(c_size)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and current_chunk_halves < self.max_chunk_halves:
                    oom_happened = True
                else:
                    raise

            if not oom_happened:
                break

            current_chunk_halves += 1
            chunk_size = max(1, chunk_size // 2)
            if self.debug:
                self.logger.debug(f"OOM occurred. Reducing chunk size to {chunk_size}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._last_effective_chunk_size = int(chunk_size)
        self._last_chunk_halves = int(current_chunk_halves)

        if not loss_chunks:
            return anchors_local.sum() * 0.0

        weighted_sum = torch.sum(torch.stack([w * lc for w, lc in zip(weight_chunks, loss_chunks)]))
        return weighted_sum / sum(weight_chunks)

    def _stream_full_denom_log_z(
        self,
        *,
        anchors: torch.Tensor,
        pool: torch.Tensor,
        positive_indices: torch.Tensor,
        self_indices: torch.Tensor | None,
        rows_per_sample: int,
        views: int,
        tau: torch.Tensor | float,
    ) -> torch.Tensor:
        """
        Stream exact per-anchor log normalizers without materializing full [C, P] logits.
        """
        C = int(anchors.size(0))
        if C <= 0:
            return anchors.new_zeros((0,), dtype=torch.float32)

        key_tile_size = min(int(self.full_denom_key_tile_size), int(pool.size(0)))
        if key_tile_size <= 0:
            raise RuntimeError("PMAContrastiveLoss full-denominator path received empty key pool.")

        # Stable running logsumexp stats: m = running max, s = running sum(exp(logit - m)).
        m = torch.full((C,), float("-inf"), device=anchors.device, dtype=torch.float32)
        s = torch.zeros((C,), device=anchors.device, dtype=torch.float32)

        for tile_start in range(0, int(pool.size(0)), key_tile_size):
            tile_end = min(tile_start + key_tile_size, int(pool.size(0)))
            key_tile = pool[tile_start:tile_end]

            tile_logits = safe_logits(anchors @ key_tile.t(), tau, debug=self.debug)
            valid_mask = self._build_streamed_full_denom_valid_mask(
                positive_indices=positive_indices,
                self_indices=self_indices,
                global_size=int(pool.size(0)),
                rows_per_sample=rows_per_sample,
                views=views,
                tile_start=tile_start,
                tile_end=tile_end,
            )

            if not bool(valid_mask.any().item()):
                continue

            minus_inf = torch.full_like(tile_logits, float("-inf"))
            masked_logits = torch.where(valid_mask, tile_logits, minus_inf)
            tile_max = masked_logits.max(dim=1).values
            new_m = torch.maximum(m, tile_max)

            prev_scale = torch.where(torch.isfinite(m), torch.exp(m - new_m), torch.zeros_like(m))
            tile_exp = torch.where(
                valid_mask, torch.exp(tile_logits - new_m.unsqueeze(1)), torch.zeros_like(tile_logits)
            )
            s = prev_scale * s + tile_exp.sum(dim=1)
            m = new_m

        if bool((s <= 0).any().item()):
            raise RuntimeError(
                "PMAContrastiveLoss streamed full denominator has anchors with no valid candidates. "
                "Check pool geometry and exclusion settings."
            )

        tiny = torch.finfo(s.dtype).tiny
        return m + torch.log(s.clamp_min(tiny))

    def _build_streamed_full_denom_valid_mask(
        self,
        *,
        positive_indices: torch.Tensor,
        self_indices: torch.Tensor | None,
        global_size: int,
        rows_per_sample: int,
        views: int,
        tile_start: int,
        tile_end: int,
    ) -> torch.Tensor:
        """
        Build [C, tile] validity mask for exact streamed denominators.
        """
        if views not in (1, 2):
            raise ValueError(f"PMAContrastiveLoss only supports views in {{1,2}}, got {views}")

        device = positive_indices.device
        C = int(positive_indices.size(0))
        tile_len = int(tile_end - tile_start)
        if tile_len <= 0:
            return torch.zeros((C, 0), dtype=torch.bool, device=device)

        tile_indices = torch.arange(tile_start, tile_end, device=device, dtype=positive_indices.dtype).unsqueeze(0)
        valid = torch.ones((C, tile_len), dtype=torch.bool, device=device)

        if self.exclude_local_sample:
            rps = max(int(rows_per_sample), 1)
            if views == 1:
                if global_size % rps != 0:
                    raise RuntimeError(
                        "PMAContrastiveLoss expects global_size divisible by rows_per_sample when excluding samples: "
                        f"global_size={global_size}, rows_per_sample={rps}"
                    )
                sample_ids = torch.div(positive_indices, rps, rounding_mode="floor")
                block_start = sample_ids * rps
                in_local = (tile_indices >= block_start.unsqueeze(1)) & (
                    tile_indices < (block_start + rps).unsqueeze(1)
                )
                valid = valid & (~in_local)
            else:
                if global_size % 2 != 0:
                    raise RuntimeError(f"Combined view pool expects even global_size, got {global_size}")
                half = global_size // 2
                if half % rps != 0:
                    raise RuntimeError(
                        "Combined view pool expects half divisible by rows_per_sample: "
                        f"half={half}, rows_per_sample={rps}"
                    )
                local_indices = positive_indices % half
                sample_ids = torch.div(local_indices, rps, rounding_mode="floor")
                block_start_first = sample_ids * rps
                block_start_second = block_start_first + half
                in_first = (tile_indices >= block_start_first.unsqueeze(1)) & (
                    tile_indices < (block_start_first + rps).unsqueeze(1)
                )
                in_second = (tile_indices >= block_start_second.unsqueeze(1)) & (
                    tile_indices < (block_start_second + rps).unsqueeze(1)
                )
                valid = valid & (~in_first) & (~in_second)
        elif views == 2:
            if self_indices is None:
                raise ValueError("self_indices must be provided when views=2 and exclude_local_sample=False")
            valid = valid & (tile_indices != self_indices.unsqueeze(1))

        # Positive key must always be included in the denominator.
        valid = valid | (tile_indices == positive_indices.unsqueeze(1))
        return valid

    def _sample_negatives_for_chunk(
        self,
        positive_indices: torch.Tensor,
        self_indices: torch.Tensor | None,
        global_size: int,
        rows_per_sample: int,
        views: int,
    ) -> torch.Tensor:
        """
        Sample per-anchor negatives without allocating a [chunk, global_size] mask.

        Args:
            positive_indices: Shape [C], the positive row indices in the key pool.
            self_indices: Shape [C] when views=2 and exclude_local_sample=False, else None.
            global_size: Total number of rows in the global tensor
            rows_per_sample: Number of rows (states) belonging to each sample
            views: Number of packed views in the pool (1 or 2)

        Returns:
            Tensor of shape [C, n_negative_samples] with global indices for
            negative samples, or None if no valid negatives could be found.
        """
        device, C = positive_indices.device, positive_indices.size(0)
        if C == 0:
            return None
        if views not in (1, 2):
            raise ValueError(f"PMAContrastiveLoss only supports views in {{1,2}}, got {views}")
        if self.n_negative_samples <= 0:
            raise ValueError(f"n_negative_samples must be > 0, got {self.n_negative_samples}")

        rps = max(int(rows_per_sample), 1)
        n_neg = int(self.n_negative_samples)
        positive_indices = positive_indices.clamp(0, max(global_size - 1, 0)).long()

        if self.negative_sampling_mode == "source_stratified":
            return self._sample_source_stratified_negatives(
                positive_indices=positive_indices,
                self_indices=self_indices,
                global_size=global_size,
                rows_per_sample=rps,
                views=views,
            )

        if self.exclude_local_sample:
            if global_size % rps != 0:
                raise RuntimeError(
                    f"PMAContrastiveLoss expects global_size divisible by rows_per_sample when excluding samples: "
                    f"global_size={global_size}, rows_per_sample={rps}"
                )
            if views == 1:
                valid_size = global_size - rps
                if valid_size <= 0:
                    if self.debug:
                        self.logger.warning("No valid negatives found - skipping chunk")
                    return None
                sample_ids = torch.div(positive_indices, rps, rounding_mode="floor")
                block_start = sample_ids * rps
                rand = torch.randint(0, valid_size, (C, n_neg), device=device, generator=self._g)
                return rand + (rand >= block_start.unsqueeze(1)) * rps

            if global_size % 2 != 0:
                raise RuntimeError(f"Combined view pool expects even global_size, got {global_size}")
            half = global_size // 2
            if half % rps != 0:
                raise RuntimeError(
                    f"Combined view pool expects half divisible by rows_per_sample: half={half}, rows_per_sample={rps}"
                )
            valid_size = global_size - (2 * rps)
            if valid_size <= 0:
                if self.debug:
                    self.logger.warning("No valid negatives found - skipping chunk")
                return None

            local_indices = positive_indices % half
            sample_ids = torch.div(local_indices, rps, rounding_mode="floor")
            block_start_first = sample_ids * rps
            block_start_second = block_start_first + half

            rand = torch.randint(0, valid_size, (C, n_neg), device=device, generator=self._g)
            neg = rand + (rand >= block_start_first.unsqueeze(1)) * rps
            neg = neg + (neg >= block_start_second.unsqueeze(1)) * rps
            return neg

        if views == 1:
            valid_size = global_size - 1
            if valid_size <= 0:
                if self.debug:
                    self.logger.warning("No valid negatives found - skipping chunk")
                return None
            rand = torch.randint(0, valid_size, (C, n_neg), device=device, generator=self._g)
            return rand + (rand >= positive_indices.unsqueeze(1)).to(rand.dtype)

        if self_indices is None:
            raise ValueError("self_indices must be provided when views=2 and exclude_local_sample=False")
        valid_size = global_size - 2
        if valid_size <= 0:
            if self.debug:
                self.logger.warning("No valid negatives found - skipping chunk")
            return None

        self_indices = self_indices.clamp(0, max(global_size - 1, 0)).long()
        low = torch.minimum(self_indices, positive_indices)
        high = torch.maximum(self_indices, positive_indices)
        rand = torch.randint(0, valid_size, (C, n_neg), device=device, generator=self._g)
        neg = rand + (rand >= low.unsqueeze(1)).to(rand.dtype)
        neg = neg + (rand >= (high - 1).unsqueeze(1)).to(rand.dtype)
        return neg

    def _sample_source_stratified_negatives(
        self,
        *,
        positive_indices: torch.Tensor,
        self_indices: torch.Tensor | None,
        global_size: int,
        rows_per_sample: int,
        views: int,
    ) -> torch.Tensor | None:
        """
        Sample a bounded number of rows from each non-anchor source sequence.

        The source unit is the original sequence, represented by a contiguous
        block of ``rows_per_sample`` flattened PMA state rows. This mode changes
        negative source balance, so it intentionally requires local-source
        exclusion and the standard one-view key pool.
        """
        del self_indices
        if not self.exclude_local_sample:
            raise RuntimeError(
                "PMAContrastiveLoss negative_sampling_mode='source_stratified' requires "
                "exclude_local_sample=True so negatives come only from other source sequences."
            )
        if views != 1:
            raise RuntimeError(
                "PMAContrastiveLoss negative_sampling_mode='source_stratified' currently supports only "
                "the standard one-view key pool. Disable enable_small_batch_combined_pool for this mode."
            )

        device, C = positive_indices.device, int(positive_indices.size(0))
        rps = max(int(rows_per_sample), 1)
        if int(global_size) % rps != 0:
            raise RuntimeError(
                "PMAContrastiveLoss expects global_size divisible by rows_per_sample for source-stratified "
                f"sampling: global_size={global_size}, rows_per_sample={rps}"
            )

        source_count = int(global_size) // rps
        valid_source_count = source_count - 1
        if valid_source_count <= 0:
            if self.debug:
                self.logger.warning("No valid source-stratified negatives found - skipping chunk")
            return None

        effective_per_source = min(int(self.negatives_per_source), rps)
        max_by_source = valid_source_count * effective_per_source
        n_neg = min(int(self.n_negative_samples), max_by_source)
        if n_neg <= 0:
            if self.debug:
                self.logger.warning("No source-stratified negatives requested - skipping chunk")
            return None

        anchor_sources = torch.div(positive_indices.long(), rps, rounding_mode="floor").clamp(0, source_count - 1)
        source_offsets = torch.arange(valid_source_count, device=device, dtype=torch.long).unsqueeze(0).expand(C, -1)
        candidate_sources = source_offsets + (source_offsets >= anchor_sources.unsqueeze(1)).to(torch.long)
        candidate_sources = candidate_sources.repeat_interleave(effective_per_source, dim=1)
        candidate_slots = torch.arange(effective_per_source, device=device, dtype=torch.long).repeat(valid_source_count)
        candidate_slots = candidate_slots.unsqueeze(0).expand(C, -1)

        candidate_count = int(candidate_sources.size(1))
        if n_neg < candidate_count:
            scores = torch.rand((C, candidate_count), device=device, generator=self._g)
            selected_positions = torch.topk(scores, k=n_neg, dim=1, largest=False, sorted=False).indices
            selected_sources = candidate_sources.gather(1, selected_positions)
            selected_slots = candidate_slots.gather(1, selected_positions)
        else:
            selected_sources = candidate_sources
            selected_slots = candidate_slots

        source_base_offsets = torch.randint(0, rps, (C, source_count), device=device, generator=self._g)
        row_offsets = (source_base_offsets.gather(1, selected_sources) + selected_slots) % rps
        return selected_sources * rps + row_offsets

    def forward(self, states1: torch.Tensor, states2: torch.Tensor) -> torch.Tensor:
        """
        Forward method for nn.Module compatibility - calls compute_loss
        """
        return self.compute_loss(states1, states2)
