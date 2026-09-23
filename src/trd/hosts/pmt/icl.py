import logging

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_f
import torch.nn as nn
import torch.nn.functional as F

from .utils import safe_logits


logger = logging.getLogger("PMT")


def _gather_row_counts(local_rows: int, device: torch.device) -> list[int]:
    """
    Gather row counts from all ranks.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [int(local_rows)]

    world_size = dist.get_world_size()
    rows_local = torch.tensor([int(local_rows)], device=device, dtype=torch.long)
    gathered = [torch.zeros_like(rows_local) for _ in range(world_size)]
    dist.all_gather(gathered, rows_local)
    row_counts = [int(t.item()) for t in gathered]
    if any(n < 0 for n in row_counts):
        raise ValueError(f"ICL gather received negative row counts: {row_counts}")
    return row_counts


def _gather_view_row_counts(
    local_rows_z1: int, local_rows_z2: int, device: torch.device
) -> tuple[list[int], list[int]]:
    """
    Gather row counts for both views in one collective.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [int(local_rows_z1)], [int(local_rows_z2)]

    world_size = dist.get_world_size()
    rows_local = torch.tensor([int(local_rows_z1), int(local_rows_z2)], device=device, dtype=torch.long)
    gathered = [torch.zeros_like(rows_local) for _ in range(world_size)]
    dist.all_gather(gathered, rows_local)
    rows_z1 = [int(t[0].item()) for t in gathered]
    rows_z2 = [int(t[1].item()) for t in gathered]
    if any(n < 0 for n in rows_z1) or any(n < 0 for n in rows_z2):
        raise ValueError(f"ICL gather received negative row counts: z1={rows_z1}, z2={rows_z2}")
    return rows_z1, rows_z2


def _all_gather_with_grad(x: torch.Tensor, row_counts: list[int] | None = None) -> tuple[torch.Tensor, int, int]:
    """
    Differentiable all-gather for 2D tensors [N_local, D], with padding for uneven N_local.
    """
    if x.dim() != 2:
        raise ValueError(f"ICL all-gather expects a 2D tensor [N, D], got shape={tuple(x.shape)}")

    if not (dist.is_available() and dist.is_initialized()):
        return x, 0, x.size(0)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if row_counts is None:
        row_counts = _gather_row_counts(x.size(0), x.device)
    if len(row_counts) != world_size:
        raise ValueError(f"ICL row_counts length mismatch: expected {world_size}, got {len(row_counts)}")
    if row_counts[rank] != x.size(0):
        raise ValueError(
            f"ICL local row count mismatch for rank {rank}: row_counts[{rank}]={row_counts[rank]} vs local={x.size(0)}"
        )

    max_rows = max(row_counts) if row_counts else 0
    if max_rows < 0:
        raise ValueError(f"ICL invalid max_rows from row_counts={row_counts}")
    if max_rows > 1_000_000_000:
        raise ValueError(f"ICL row_counts unrealistically large: {row_counts}")

    if x.size(0) < max_rows:
        pad = x.new_zeros((max_rows - x.size(0), x.size(1)))
        x_pad = torch.cat([x, pad], dim=0)
    else:
        x_pad = x

    gathered = dist_nn_f.all_gather(x_pad)  # Tuple[Tensor, ...], each [max_rows, D]
    chunks = [g[:n] for g, n in zip(gathered, row_counts) if n > 0]
    global_x = torch.cat(chunks, dim=0) if chunks else x.new_zeros((0, x.size(1)))
    offset = int(sum(row_counts[:rank]))
    total = int(sum(row_counts))
    return global_x, offset, total


def compute_contrastive_loss_icl(
    cls_embeddings1: torch.Tensor,
    cls_embeddings2: torch.Tensor,
    temperature: float | torch.Tensor = 0.1,
    use_all_gather: bool = False,
    use_local_anchors: bool = False,
    logger: logging.Logger | None = None,
    diagnostics: dict[str, float] | None = None,
) -> torch.Tensor:
    """
    Compute bidirectional contrastive loss between class tokens from two augmented views.
    In distributed mode, all-gather uses an autograd-capable path.

    Args:
        cls_embeddings1: shape [batch_size, d_model]
        cls_embeddings2: shape [batch_size, d_model]
        temperature: scaling factor
        use_all_gather: if True, gather embeddings across ranks for global negatives.
                        If False, do local negatives only.
        use_local_anchors: backward-compatible flag. Anchors are always local in distributed mode.
        logger: optional logger
        diagnostics: optional mutable mapping populated with the realized candidate geometry.
    """
    logger = logger or logging.getLogger("PMT")
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    if cls_embeddings1.shape != cls_embeddings2.shape:
        raise ValueError(
            f"ICL contrastive expects equal shapes, got {cls_embeddings1.shape} vs {cls_embeddings2.shape}"
        )
    z1_local = F.normalize(cls_embeddings1, p=2, dim=-1, eps=1e-6)
    z2_local = F.normalize(cls_embeddings2, p=2, dim=-1, eps=1e-6)

    local_batch = z1_local.size(0)

    if use_all_gather and is_distributed:
        if not use_local_anchors and rank == 0:
            logger.debug(
                "[ICL] use_local_anchors=False is deprecated in distributed mode; "
                "using local anchors with differentiable global pools."
            )

        rows_z1, rows_z2 = _gather_view_row_counts(local_batch, z2_local.size(0), z1_local.device)
        if rows_z1 != rows_z2:
            raise ValueError(
                f"ICL contrastive expects per-rank row counts to match between views, got z1={rows_z1}, z2={rows_z2}"
            )

        z1_global, z1_offset, global_batch_z1 = _all_gather_with_grad(z1_local, rows_z1)
        z2_global, z2_offset, global_batch_z2 = _all_gather_with_grad(z2_local, rows_z2)

        if global_batch_z1 != global_batch_z2:
            raise ValueError(
                f"ICL contrastive expects equal global batch sizes for both views, got z1={global_batch_z1}, z2={global_batch_z2}"
            )
        global_batch = global_batch_z1

        if (z1_offset + local_batch > global_batch) or (z2_offset + local_batch > global_batch):
            raise ValueError(
                f"ICL offsets out of range: z1_offset={z1_offset}, z2_offset={z2_offset}, "
                f"local_batch={local_batch}, global_batch={global_batch}"
            )
    else:
        z1_global, z2_global = z1_local, z2_local
        z1_offset = z2_offset = 0
        global_batch = local_batch

    if diagnostics is not None:
        all_gather_active = bool(use_all_gather and is_distributed)
        diagnostics.clear()
        diagnostics.update(
            {
                "icl_actual_local_batch_size": float(local_batch),
                "icl_candidate_pool_size": float(global_batch),
                "icl_effective_negatives_per_anchor": float(max(global_batch - 1, 0)),
                "icl_zero_negative_batch": float(global_batch <= 1),
                "icl_all_gather_active": float(all_gather_active),
                "icl_memory_bank_active": 0.0,
            }
        )

    if global_batch == 0 or local_batch == 0:
        loss = (z1_local.sum() + z2_local.sum()) * 0.0
    else:
        pos_indices_12 = torch.arange(local_batch, device=z1_local.device) + z2_offset
        logits_12 = safe_logits(torch.matmul(z1_local, z2_global.T), temperature)
        loss_i = F.cross_entropy(logits_12, pos_indices_12)

        pos_indices_21 = torch.arange(local_batch, device=z2_local.device) + z1_offset
        logits_21 = safe_logits(torch.matmul(z2_local, z1_global.T), temperature)
        loss_j = F.cross_entropy(logits_21, pos_indices_21)

        loss = 0.5 * (loss_i + loss_j)

    if is_distributed:
        # Weighted average keeps logging accurate under uneven local batch sizes.
        stats = torch.tensor(
            [float(loss.detach()) * float(local_batch), float(local_batch)], device=loss.device, dtype=torch.float32
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if rank == 0:
            avg_loss = (stats[0] / stats[1]).item() if stats[1].item() > 0 else 0.0
            logger.debug(
                f"ICL loss (weighted avg): {avg_loss:.4f} "
                f"(local_batch={local_batch}, global_batch={global_batch}, use_all_gather={use_all_gather})"
            )
    elif rank == 0:
        logger.debug(f"ICL loss: {loss.item():.4f} (batch_size={global_batch}, use_all_gather={use_all_gather})")

    if rank == 0:
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error(f"Invalid loss value: {loss.item()}")
            raise ValueError("Contrastive loss is NaN/Inf.")

    return loss


class InstanceContrastiveLoss(nn.Module):
    """
    Instance contrastive loss over CLS tokens with temperature scheduling support.
    This is a thin wrapper around compute_contrastive_loss_icl that adds
    temperature scheduling capability.
    """

    def __init__(
        self,
        temperature: float,
        end_temperature: float,
        schedule_epochs: int,
        temp_schedule_mode: str = "absolute",
        use_all_gather: bool = False,
        use_local_anchors: bool = False,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the ICL contrastive loss with temperature scheduling.

        Args:
            temperature: Starting temperature value
            end_temperature: Final temperature value after schedule_epochs
            schedule_epochs: Number of epochs over which to linearly decay the temperature
            use_all_gather: Whether to gather tensors across distributed processes
            logger: Optional logger for debug messages
        """
        super().__init__()

        self.register_buffer("_tau", torch.tensor(float(temperature)))
        self._τ0 = temperature
        if temp_schedule_mode == "fraction":
            self._τ1 = temperature * end_temperature
        else:
            self._τ1 = end_temperature
        self._sched_len = schedule_epochs
        self._current_epoch = 0
        self.use_all_gather = use_all_gather
        self.use_local_anchors = use_local_anchors
        self.logger = logger or logging.getLogger("PMT")
        self._last_diagnostics: dict[str, float] = {}

    @torch.no_grad()
    def set_epoch(self, epoch: int):
        """
        Update the temperature based on the current epoch.

        Args:
            epoch: Current training epoch
        """
        self._current_epoch = epoch

        if self._sched_len > 0:
            t = min(epoch / self._sched_len, 1.0)
            new_temp = self._τ0 + t * (self._τ1 - self._τ0)
            self._tau.fill_(new_temp)

    def forward(self, cls_embeddings1: torch.Tensor, cls_embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss using the current temperature.

        Args:
            cls_embeddings1: CLS token embeddings from view 1
            cls_embeddings2: CLS token embeddings from view 2

        Returns:
            Contrastive loss
        """
        return compute_contrastive_loss_icl(
            cls_embeddings1,
            cls_embeddings2,
            temperature=self._tau,
            use_all_gather=self.use_all_gather,
            use_local_anchors=self.use_local_anchors,
            logger=self.logger,
            diagnostics=self._last_diagnostics,
        )

    def update_temperature(self, epoch: int):
        """Public helper so external callers/tests can update τ without using the internal name."""
        self.set_epoch(epoch)
