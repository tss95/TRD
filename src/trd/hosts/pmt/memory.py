import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from trd.hosts.pmt.transformer import choose_norm, sanitise_flash_mask
from typing import Tuple, Optional, Dict

import numpy as np  # noqa: F401

# Import shared components
from trd.hosts.pmt.components import (
    get_gate,
    LocalOffsetEmbedding,
    ChunkOffsetEmbedding,
    SkipGate,
    push_gate_stats,
    run_with_optional_window_checkpoint,
    should_use_window_checkpointing,
)

logger = logging.getLogger('PMT')


"""
Shifted Progressive Memory Attention (sPMA) - Version 5.4 (bootstrap)

This module is intentionally bootstrapped from PMA v5.3 as a starting point
for sPMA development. Behavior is currently expected to match v5.3 until
windowing/routing changes are introduced incrementally.
"""


# -------------------------------------------------------------------
#  GateModule for mixing previous and random states
# -------------------------------------------------------------------
# GateModule and get_gate are now imported from trd.hosts.pmt.components


# -------------------------------------------------------------------
#  Position Encoding Components
# -------------------------------------------------------------------
# LocalOffsetEmbedding and ChunkOffsetEmbedding are now imported from trd.hosts.pmt.components


# -------------------------------------------------------------------
#  PMAChunkProcessor - Sliding window encoder with state tokens
# -------------------------------------------------------------------
class PMAChunkProcessor(nn.Module):
    """Sliding window encoder with recurrent state tokens.

    Processes input sequences using overlapping windows with state tokens
    that carry information between windows. Supports both RoPE and learned
    position encodings.

    Args:
        d_model: Hidden dimension
        num_heads: Number of attention heads
        d_ff: Feedforward dimension
        n_states: Number of state tokens
        window_size: Size of each window
        window_stride: Stride between windows
        activation: Activation function ('relu' or 'gelu')
        dropout: Dropout probability
        masking: Masking strategy ('causal' or None)
        use_rope: Whether to use rotary position embeddings
        rope_chunk_offset_on_states: Keep chunk offsets active for state tokens when use_rope=True
        rope_base: Base for RoPE frequencies
        gate_type: Type of state mixing gate
        experiment_mode: Experimental mode for ablations
        dropout_mode: Dropout mode for attention
        state_update_gate_enabled: Enable horizontal state update gate (v5.3)
        state_update_gate_kind: Gate type for horizontal update ('scalar' or 'dimwise')
        state_update_gate_init_logit: Initial logit for update gate (negative favors copy)
        state_update_gate_temperature: Temperature for update gate sigmoid
        state_read_enabled: Enable state -> token read injection (v5.3)
        state_read_kind: Gate type for state read ('scalar' or 'dimwise')
        state_read_init_logit: Initial logit for state read gate (negative favors off)
        state_read_temperature: Temperature for state read gate sigmoid
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        n_states: int,
        window_size: int,
        window_stride: int,
        activation='relu',
        dropout=0.1,
        masking='causal',
        use_rope=False,
        rope_base=10000.0,
        local_offset_scale=0.1,
        use_chunk_offset=False,
        rope_chunk_offset_on_states: bool = False,
        max_chunks=10000,
        experiment_mode='pmt',
        chunk_offset_scale=1.0,
        norm_type="ln",
        norm_position="pre",
        debug=False,
        name="PMAChunkProcessor",
        gate_type=None,
        dropout_mode="legacy",
        gate_temperature=0.5,
        gate_use_residual=True,
        gate_init_logit=None,
        use_horizontal_state_passing: bool = True,
        allow_state_to_attend_prev_state: bool = False,
        state_init_noise_std: float = 0.01,
        window_checkpointing: bool = False,
        state_update_gate_enabled: bool = False,
        state_update_gate_kind: str = "dimwise",
        state_update_gate_init_logit: float = -2.0,
        state_update_gate_temperature: float = 1.0,
        state_read_enabled: bool = False,
        state_read_kind: str = "dimwise",
        state_read_init_logit: float = -2.0,
        state_read_temperature: float = 1.0,
        compile_per_window: bool = False,
        compile_mode: str = "reduce-overhead",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_states = n_states
        self.window_size = window_size
        # sPMA v5.4 uses strict non-overlapping windows within each block.
        self.window_stride = window_size
        if window_stride != window_size and debug:
            logger.info(
                "%s: overriding window_stride=%s to window_size=%s for sPMA non-overlap geometry.",
                name,
                window_stride,
                window_size,
            )
        self.debug = debug
        self.name = name
        self.experiment_mode = experiment_mode
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.masking = masking
        self.norm_position = norm_position.lower()
        self.use_chunk_offset = use_chunk_offset
        self.rope_chunk_offset_on_states = bool(rope_chunk_offset_on_states)
        self.gate_type = gate_type
        self.use_horizontal_state_passing = use_horizontal_state_passing
        self.allow_state_to_attend_prev_state = allow_state_to_attend_prev_state
        self.state_init_noise_std = float(state_init_noise_std)
        self.window_checkpointing = bool(window_checkpointing)
        self.compile_per_window = bool(compile_per_window)
        self.compile_mode = str(compile_mode or "reduce-overhead")
        self._compiled_process_window_prepared = {}
        self._compiled_batch_size_by_variant = {}
        self._warned_compile_batch_size_fallback = False
        self._rope_compile_cache_warmed = False
        self.state_update_gate_temperature = float(state_update_gate_temperature)
        self.state_read_temperature = float(state_read_temperature)

        # State tokens
        self.sos_state_tokens = nn.Parameter(torch.randn(n_states, d_model) * 0.02)
        # Learned base current-state initialization (replaces content-hash seeding)
        self.base_state = nn.Parameter(torch.randn(n_states, d_model) * 0.02)
        self.state_norm = choose_norm(norm_type, d_model)

        # State mixing gate with improved activation and formulation
        if gate_type is not None:
            self.gate = get_gate(
                gate_type,
                d_model,
                n_states,
                temperature=gate_temperature,
                use_residual=gate_use_residual,
                init_logit=gate_init_logit,
            )
        else:
            self.gate = None

        # Horizontal state update gate (Phase 1.1)
        self.state_update_gate = None
        if state_update_gate_enabled and n_states > 0:
            gate_kind = str(state_update_gate_kind).lower()
            if gate_kind not in {"scalar", "dimwise"}:
                raise ValueError(
                    f"state_update_gate_kind must be 'scalar' or 'dimwise', got '{state_update_gate_kind}'"
                )
            gate_out = 1 if gate_kind == "scalar" else d_model
            self.state_update_gate = nn.Linear(3 * d_model, gate_out, bias=True)
            nn.init.zeros_(self.state_update_gate.weight)
            nn.init.constant_(self.state_update_gate.bias, float(state_update_gate_init_logit))
            self.state_update_gate_kind = gate_kind

        # State -> token read gate (Phase 1.2)
        self.state_read_gate = None
        self.state_read_proj = None
        if state_read_enabled and n_states > 0:
            read_kind = str(state_read_kind).lower()
            if read_kind not in {"scalar", "dimwise"}:
                raise ValueError(f"state_read_kind must be 'scalar' or 'dimwise', got '{state_read_kind}'")
            gate_out = 1 if read_kind == "scalar" else d_model
            self.state_read_gate = nn.Linear(d_model, gate_out, bias=True)
            nn.init.zeros_(self.state_read_gate.weight)
            nn.init.constant_(self.state_read_gate.bias, float(state_read_init_logit))
            self.state_read_proj = nn.Linear(d_model, d_model, bias=True)
            nn.init.zeros_(self.state_read_proj.weight)
            nn.init.zeros_(self.state_read_proj.bias)
            self.state_read_kind = read_kind

        # Token-local offsets are additive-only (non-RoPE). Chunk offsets can
        # optionally remain active for state tokens under RoPE.
        self.local_offset_embed = (
            LocalOffsetEmbedding(window_size, d_model, local_offset_scale) if not use_rope else None
        )
        chunk_offset_enabled = bool(use_chunk_offset and (not use_rope or self.rope_chunk_offset_on_states))
        self.chunk_offset_embed = (
            ChunkOffsetEmbedding(max_chunks, d_model, chunk_offset_scale) if chunk_offset_enabled else None
        )

        # Transformer layer
        from trd.hosts.pmt.transformer import TransformerEncoderLayerWithMask

        # Dynamo guards on the layer debug name because TransformerEncoderLayerWithMask
        # passes it through f-strings in finite-check messages. Keep compiled v5.4
        # windows monomorphic across PMA blocks; eager keeps the block-specific name.
        layer_name = "SPMACompiledWindow_MainLayer" if self.compile_per_window else f"{name}_MainLayer"
        self.layer = TransformerEncoderLayerWithMask(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
            norm_position=norm_position,
            name=layer_name,
            dropout_mode=dropout_mode,
        )

        # Build causal mask
        self.total_length = 2 * n_states + window_size
        chunk_mask = self._build_mask(self.total_length, window_size, masking)
        self.register_buffer("mask", chunk_mask, persistent=False)

    def _build_mask(self, total_len: int, window_size: int, masking: str = 'causal') -> torch.BoolTensor:
        """Build attention mask for chunk processing."""
        n_s = self.n_states
        _, C, T0 = 0, n_s, 2 * n_s
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)

        if masking == 'causal':
            # Window queries cannot see future window tokens
            rows = torch.arange(T0, T0 + window_size)[:, None]
            cols = torch.arange(T0, T0 + window_size)[None, :]
            mask[T0 : T0 + window_size, T0 : T0 + window_size] = cols > rows
            # Window queries cannot see current state block
            mask[T0 : T0 + window_size, C : C + n_s] = True

        # Optionally block current state queries from attending to previous states
        if not self.allow_state_to_attend_prev_state:
            mask[C : C + n_s, 0:C] = True

        return mask

    def _prepare_windows(self, x, patch_mask=None, *, window_offset: int = 0):
        """Extract non-overlapping windows with optional left shift offset."""
        B, L, D = x.shape
        device = x.device

        if window_offset < 0 or window_offset >= self.window_size:
            raise ValueError(f"{self.name}: window_offset must be in [0, {self.window_size - 1}], got {window_offset}.")

        # Shift by left-padding `window_offset`, then right-pad to the window multiple.
        front_pad = int(window_offset)
        total_unpadded = L + front_pad
        end_pad = (self.window_size - (total_unpadded % self.window_size)) % self.window_size
        Lp = total_unpadded + end_pad

        x_pad = F.pad(x, (0, 0, front_pad, end_pad))
        B, Lp, D = x_pad.shape

        num_win = Lp // self.window_size
        if num_win <= 0:
            raise ValueError("No valid windows with given window_size")

        # Extract disjoint windows.
        windows = x_pad.view(B, num_win, self.window_size, D).contiguous()
        if __debug__:
            assert windows.shape == (B, num_win, self.window_size, D)

        # Per-window validity flags from the input mask, padded exactly like `x`.
        #
        # `patch_invalid_windows[b, w, k]` is True when the real-token slot `k`
        # of window `w` for batch element `b` is INVALID (must be blocked in
        # chunk attention). We mark the geometry-fill region (front/end pad) as
        # invalid as well so the two effects compose, but geometry padding is
        # still handled independently by `window_pad_indicator`/`padded_rows`.
        #
        # Kept as None when there is nothing to mask so the all-valid / None
        # path stays byte-identical to the pre-mask behaviour (no batched mask
        # is ever built, no recompile of the per-window body is triggered).
        patch_invalid_windows = None
        if patch_mask is not None:
            pm = patch_mask
            if pm.dtype != torch.bool:
                pm = pm.bool()
            if pm.size(1) > L:
                pm = pm[:, :L]
            elif pm.size(1) < L:
                pm = F.pad(pm, (0, L - pm.size(1)), value=True)
            # Only do real work if some position is actually invalid.
            if bool((~pm).any()):
                # invalid = True; pad the validity mask the same way as x.
                invalid_unpadded = ~pm  # [B, L], True = invalid
                invalid_padded = F.pad(invalid_unpadded, (front_pad, end_pad), value=True)
                patch_invalid_windows = invalid_padded.view(B, num_win, self.window_size).contiguous()

        # Add local offset embedding
        if not self.use_rope and self.local_offset_embed is not None:
            idx = torch.arange(self.window_size, device=device)
            local_emb = self.local_offset_embed(idx).unsqueeze(0).unsqueeze(0)
            windows = windows + local_emb

        # Prepare RoPE positions if needed
        absolute_positions = None
        if self.use_rope:
            window_start_indices = torch.arange(0, Lp, self.window_size, device=device)
            token_indices_in_window = torch.arange(self.window_size, device=device)
            pos_padded = window_start_indices[:, None] + token_indices_in_window[None, :]
            pos_orig = pos_padded - front_pad
            # Keep RoPE indices stable in original coordinates across offset schedules.
            absolute_positions = pos_orig.masked_fill((pos_orig < 0) | (pos_orig >= L), -1)
            positions_for_windows = torch.full((num_win, self.total_length), -1, device=device, dtype=torch.long)
            positions_for_windows[:, 2 * self.n_states :] = absolute_positions
        else:
            positions_for_windows = None

        return {
            'windows': windows,
            'absolute_positions': absolute_positions,
            'positions_for_windows': positions_for_windows,
            'num_win': num_win,
            'front_pad': front_pad,
            'end_pad': end_pad,
            'Lp': Lp,
            'L': L,
            'device': device,
            'window_offset': int(window_offset),
            'patch_invalid_windows': patch_invalid_windows,
        }

    def _route_window_states(
        self, prev_all_states: torch.Tensor, *, prev_offset: int, cur_offset: int, num_win_cur: int
    ) -> torch.Tensor:
        """Route per-window states across offset changes.

        Routing follows overlap-weighted mixing between neighboring windows
        when offsets differ (e.g., 0 <-> W//2) and also handles N_prev != N_cur.
        """
        if prev_all_states.ndim != 4:
            raise ValueError(
                f"{self.name}: prev_all_states must have shape [B, N_prev, n_states, D], got {tuple(prev_all_states.shape)}"
            )
        if num_win_cur <= 0:
            raise ValueError(f"{self.name}: num_win_cur must be positive, got {num_win_cur}")
        if prev_offset < 0 or prev_offset >= self.window_size:
            raise ValueError(f"{self.name}: prev_offset must be in [0, {self.window_size - 1}], got {prev_offset}")
        if cur_offset < 0 or cur_offset >= self.window_size:
            raise ValueError(f"{self.name}: cur_offset must be in [0, {self.window_size - 1}], got {cur_offset}")

        B, n_prev, _, _ = prev_all_states.shape
        device = prev_all_states.device
        dtype = prev_all_states.dtype

        # Sentinel states for out-of-range neighbors at boundaries.
        left_sos = self.sos_state_tokens.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        right_pad = torch.zeros_like(left_sos)
        state_bank = torch.cat([left_sos, prev_all_states, right_pad], dim=1)  # [B, N_prev + 2, nS, D]
        max_bank_idx = n_prev + 1

        j = torch.arange(num_win_cur, device=device, dtype=torch.long)
        delta = int(cur_offset) - int(prev_offset)
        alpha = float(min(1.0, max(0.0, abs(delta) / float(max(1, self.window_size)))))

        if delta == 0:
            same_idx = torch.clamp(j + 1, 0, max_bank_idx)
            return state_bank.index_select(1, same_idx)

        if delta > 0:
            # 0 -> shift: S_new[j] = alpha * S_old[j-1] + (1-alpha) * S_old[j]
            left_idx = torch.clamp(j, 0, max_bank_idx)
            right_idx = torch.clamp(j + 1, 0, max_bank_idx)
            left_states = state_bank.index_select(1, left_idx)
            right_states = state_bank.index_select(1, right_idx)
            return alpha * left_states + (1.0 - alpha) * right_states

        # shift -> 0: S_new[j] = (1-alpha) * S_old[j] + alpha * S_old[j+1]
        left_idx = torch.clamp(j + 1, 0, max_bank_idx)
        right_idx = torch.clamp(j + 2, 0, max_bank_idx)
        left_states = state_bank.index_select(1, left_idx)
        right_states = state_bank.index_select(1, right_idx)
        return (1.0 - alpha) * left_states + alpha * right_states

    def _get_base_attention_masks(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached unsanitized and sanitized PMA masks on the requested device."""
        if not hasattr(self, 'base_mask_on_device') or self.base_mask_on_device.device != device:
            self.base_mask_on_device = self.mask.to(device)
            self.base_mask_sanitized_on_device = sanitise_flash_mask(self.base_mask_on_device, policy="self_attn")
        elif not hasattr(self, 'base_mask_sanitized_on_device') or self.base_mask_sanitized_on_device.device != device:
            self.base_mask_sanitized_on_device = sanitise_flash_mask(self.base_mask_on_device, policy="self_attn")
        return self.base_mask_on_device, self.base_mask_sanitized_on_device

    def _prepare_window_runtime(
        self,
        *,
        w_i: int,
        prev_block_states: Optional[torch.Tensor],
        prev_block_all_states: Optional[torch.Tensor],
        front_pad: int,
        seq_len: int,
        positions_for_windows: Optional[torch.Tensor],
        absolute_positions: Optional[torch.Tensor],
        device: torch.device,
        patch_invalid_win: Optional[torch.Tensor] = None,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        bool,
    ]:
        """Prepare Python-indexed per-window inputs before the compiled tensor body.

        ``patch_invalid_win`` is an optional ``[B, window_size]`` boolean tensor
        where True marks a real-token slot that is INVALID under the input
        validity mask (``patch_mask``) and must be excluded from chunk
        attention. When supplied (and any slot is actually invalid), the
        returned ``chunk_attn_mask`` becomes a batched ``[B, total_length,
        total_length]`` mask carrying both the existing window-fill geometry
        padding and the per-sample patch-invalid rows/columns. When it is None
        (or all-valid), the mask path is byte-identical to the pre-mask
        behaviour so existing results are unaffected.
        """
        prev_state_source = None
        if prev_block_all_states is not None and w_i < prev_block_all_states.shape[1]:
            prev_state_source = prev_block_all_states[:, w_i]
        elif prev_block_states is not None:
            prev_state_source = prev_block_states

        chunk_offset_vec = None
        if self.use_chunk_offset and self.chunk_offset_embed is not None:
            chunk_offset_vec = self.chunk_offset_embed(w_i, device).unsqueeze(0)

        positions_tensor = None
        if self.use_rope:
            if positions_for_windows is not None:
                positions_tensor = positions_for_windows[w_i]
            elif absolute_positions is not None:
                positions_tensor = torch.full((self.total_length,), -1, device=device, dtype=torch.long)
                positions_tensor[2 * self.n_states :] = absolute_positions[w_i]

        base_mask, base_mask_sanitized = self._get_base_attention_masks(device)
        chunk_attn_mask = base_mask_sanitized

        # Apply padding mask using integer bounds to avoid device-scalar reductions.
        valid_start = front_pad
        valid_end = front_pad + seq_len
        win_start = w_i * self.window_size
        win_end = win_start + self.window_size

        window_pad_indicator = None
        padded_rows = None
        if not (win_start >= valid_start and win_end <= valid_end):
            chunk_attn_mask = base_mask.clone()
            window_positions = torch.arange(win_start, win_end, device=device)
            window_pad_indicator = (window_positions >= valid_start) & (window_positions < valid_end)
            padded_rows = torch.zeros(self.total_length, dtype=torch.bool, device=device)
            padded_rows[2 * self.n_states :] = ~window_pad_indicator
            padded_row_or_col = padded_rows.unsqueeze(1) | padded_rows.unsqueeze(0)
            chunk_attn_mask = chunk_attn_mask | padded_row_or_col
            diag_idx = torch.arange(self.total_length, device=device)
            chunk_attn_mask[diag_idx, diag_idx] = chunk_attn_mask[diag_idx, diag_idx] & (~padded_rows)
            chunk_attn_mask = sanitise_flash_mask(chunk_attn_mask, policy="self_attn")

        # Apply the per-sample input validity mask (patch_mask) to chunk attention.
        #
        # This is intentionally kept separate from the window-fill geometry
        # padding above: geometry padding marks fill slots (zeroed in the
        # output and excluded from state gating), whereas patch-invalid slots
        # are real positions we merely hide as attention KEYS/QUERIES. We only
        # block their attention interactions; their per-token outputs are left
        # untouched (the model still produces a representation at those slots,
        # which TS2Vec-style masked reconstruction relies on).
        if patch_invalid_win is not None:
            patch_invalid_win = patch_invalid_win.to(device=device, dtype=torch.bool)
            # `patch_invalid_win` is [B, window_size]; only build the (heavier)
            # batched mask when at least one slot in this window is invalid.
            if bool(patch_invalid_win.any()):
                B = patch_invalid_win.size(0)
                # Per-sample row/col validity over the full [2*n_states + W] slots.
                # State slots ([:2*n_states]) are never patch-invalid.
                invalid_rows = torch.zeros(B, self.total_length, dtype=torch.bool, device=device)
                invalid_rows[:, 2 * self.n_states :] = patch_invalid_win  # [B, total_length]

                # Start from the UNSANITISED 2-D base mask so geometry-pad and
                # patch-invalid compose cleanly under a single sanitiser pass.
                combined = base_mask.unsqueeze(0).expand(B, -1, -1).clone()  # [B, T, T]
                if padded_rows is not None:
                    geom = padded_rows.view(1, self.total_length)
                    invalid_rows = invalid_rows | geom  # fold geometry padding in
                invalid_row_or_col = invalid_rows.unsqueeze(2) | invalid_rows.unsqueeze(1)  # [B, T, T]
                combined = combined | invalid_row_or_col
                # Diagonal safety: never fully block a query row (keep self-key)
                # so softmax has at least one legal key and cannot produce NaNs.
                diag_idx = torch.arange(self.total_length, device=device)
                combined[:, diag_idx, diag_idx] = combined[:, diag_idx, diag_idx] & (~invalid_rows)
                chunk_attn_mask = sanitise_flash_mask(combined, policy="self_attn")

        return (
            prev_state_source,
            chunk_offset_vec,
            positions_tensor,
            chunk_attn_mask,
            window_pad_indicator,
            padded_rows,
            padded_rows is not None,
        )

    def _warm_rope_compile_cache(self) -> None:
        """Populate RoPE's Python cache outside Dynamo before compiling."""
        if self._rope_compile_cache_warmed or not self.use_rope:
            return
        self._rope_compile_cache_warmed = True
        try:
            from trd.hosts.pmt.transformer import _get_rope_inv_freq

            d_head = int(getattr(getattr(self.layer, "self_attn", None), "d_head"))
            if d_head <= 0 or d_head % 2 != 0:
                return
            device = next(self.parameters()).device
            _get_rope_inv_freq(d_head // 2, float(self.rope_base), device)
        except Exception:
            # This is only a compile hygiene pre-warm; the eager RoPE path still
            # owns correctness and error reporting.
            return

    def _compiled_target_for_variant(self, *, has_padding: bool, has_prev_source: bool):
        """Return the wrapper method with a monomorphic Dynamo signature."""
        if has_padding and has_prev_source:
            return self._process_window_prepared_with_padding_with_source
        if has_padding:
            return self._process_window_prepared_with_padding_no_source
        if has_prev_source:
            return self._process_window_prepared_no_padding_with_source
        return self._process_window_prepared_no_padding_no_source

    def _ensure_compiled_process_window_prepared(self, *, has_padding: bool, has_prev_source: bool):
        """Lazily compile the tensor-only per-window body for one stable variant."""
        key = (bool(has_padding), bool(has_prev_source))
        compiled = self._compiled_process_window_prepared.get(key)
        if compiled is None:
            if not hasattr(torch, "compile"):
                raise RuntimeError(f"{self.name}: torch.compile is unavailable; disable compile_per_window.")
            # TransformerEncoderLayerWithMask emits two debug log calls on first
            # RoPE forwards. Logger calls force Dynamo graph breaks, so skip
            # those side-effect-only messages when this opt-in path is active.
            if hasattr(self.layer, "_debug_forward_count"):
                self.layer._debug_forward_count = max(int(self.layer._debug_forward_count), 2)
            self._warm_rope_compile_cache()
            target = self._compiled_target_for_variant(has_padding=has_padding, has_prev_source=has_prev_source)
            compiled = torch.compile(target, mode=self.compile_mode, fullgraph=False)
            self._compiled_process_window_prepared[key] = compiled
        return compiled

    def _clone_compiled_window_result(self, result):
        """Clone compiled outputs before they re-enter recurrent state flow."""
        window_out, prev_states, window_gate_stats = result
        return window_out.clone(), prev_states.clone(), window_gate_stats

    def _run_process_window_prepared(
        self,
        win_tok,
        rand_states,
        prev_states,
        prev_state_source,
        chunk_offset_vec,
        positions_tensor,
        chunk_attn_mask,
        window_pad_indicator,
        padded_rows,
        has_padding: bool,
        alpha,
        collect_gate_stats: bool = False,
        token_keep_win: Optional[torch.Tensor] = None,
    ):
        """Run the prepared per-window body through eager or compiled execution."""
        has_padding = bool(has_padding)
        has_prev_source = prev_state_source is not None
        # When the input validity mask (patch_mask) hides real tokens in this
        # window, route through the eager body so the state-gate token mean can
        # exclude those slots. This keeps the compiled per-window variant
        # signatures byte-identical for the common (no patch-mask) path and
        # mirrors the existing eager fallback used for partial batch sizes.
        if token_keep_win is not None:
            return self._run_process_window_prepared_eager(
                win_tok,
                rand_states,
                prev_states,
                prev_state_source,
                chunk_offset_vec,
                positions_tensor,
                chunk_attn_mask,
                window_pad_indicator,
                padded_rows,
                has_padding,
                alpha,
                collect_gate_stats=collect_gate_stats,
                token_keep_win=token_keep_win,
            )
        use_compiled_window = self.compile_per_window and win_tok.is_cuda and not collect_gate_stats
        if use_compiled_window:
            variant_key = (has_padding, has_prev_source)
            batch_size = int(win_tok.size(0))
            compiled_batch_size = self._compiled_batch_size_by_variant.setdefault(variant_key, batch_size)
            if compiled_batch_size != batch_size:
                if not self._warned_compile_batch_size_fallback:
                    logger.warning(
                        "%s: compile_per_window falling back to eager for batch_size=%s; compiled batch_size=%s. "
                        "This avoids DDP/Inductor recompiles on partial batches.",
                        self.name,
                        batch_size,
                        compiled_batch_size,
                    )
                    self._warned_compile_batch_size_fallback = True
                return self._run_process_window_prepared_eager(
                    win_tok,
                    rand_states,
                    prev_states,
                    prev_state_source,
                    chunk_offset_vec,
                    positions_tensor,
                    chunk_attn_mask,
                    window_pad_indicator,
                    padded_rows,
                    has_padding,
                    alpha,
                    collect_gate_stats=False,
                )
            compiled = self._ensure_compiled_process_window_prepared(
                has_padding=has_padding, has_prev_source=has_prev_source
            )
            if has_padding and has_prev_source:
                return self._clone_compiled_window_result(
                    compiled(
                        win_tok,
                        rand_states,
                        prev_states,
                        prev_state_source,
                        chunk_offset_vec,
                        positions_tensor,
                        chunk_attn_mask,
                        window_pad_indicator,
                        padded_rows,
                        alpha,
                        collect_gate_stats=False,
                    )
                )
            if has_padding:
                return self._clone_compiled_window_result(
                    compiled(
                        win_tok,
                        rand_states,
                        prev_states,
                        chunk_offset_vec,
                        positions_tensor,
                        chunk_attn_mask,
                        window_pad_indicator,
                        padded_rows,
                        alpha,
                        collect_gate_stats=False,
                    )
                )
            if has_prev_source:
                return self._clone_compiled_window_result(
                    compiled(
                        win_tok,
                        rand_states,
                        prev_states,
                        prev_state_source,
                        chunk_offset_vec,
                        positions_tensor,
                        chunk_attn_mask,
                        alpha,
                        collect_gate_stats=False,
                    )
                )
            return self._clone_compiled_window_result(
                compiled(
                    win_tok,
                    rand_states,
                    prev_states,
                    chunk_offset_vec,
                    positions_tensor,
                    chunk_attn_mask,
                    alpha,
                    collect_gate_stats=False,
                )
            )
        return self._run_process_window_prepared_eager(
            win_tok,
            rand_states,
            prev_states,
            prev_state_source,
            chunk_offset_vec,
            positions_tensor,
            chunk_attn_mask,
            window_pad_indicator,
            padded_rows,
            has_padding,
            alpha,
            collect_gate_stats=collect_gate_stats,
            token_keep_win=token_keep_win,
        )

    def _run_process_window_prepared_eager(
        self,
        win_tok,
        rand_states,
        prev_states,
        prev_state_source,
        chunk_offset_vec,
        positions_tensor,
        chunk_attn_mask,
        window_pad_indicator,
        padded_rows,
        has_padding: bool,
        alpha,
        collect_gate_stats: bool = False,
        token_keep_win: Optional[torch.Tensor] = None,
    ):
        """Run one prepared window through the eager padding/source variant."""
        has_prev_source = prev_state_source is not None
        if has_padding:
            if has_prev_source:
                return self._process_window_prepared_with_padding_with_source(
                    win_tok,
                    rand_states,
                    prev_states,
                    prev_state_source,
                    chunk_offset_vec,
                    positions_tensor,
                    chunk_attn_mask,
                    window_pad_indicator,
                    padded_rows,
                    alpha,
                    collect_gate_stats=collect_gate_stats,
                    token_keep_win=token_keep_win,
                )
            return self._process_window_prepared_with_padding_no_source(
                win_tok,
                rand_states,
                prev_states,
                chunk_offset_vec,
                positions_tensor,
                chunk_attn_mask,
                window_pad_indicator,
                padded_rows,
                alpha,
                collect_gate_stats=collect_gate_stats,
                token_keep_win=token_keep_win,
            )
        if has_prev_source:
            return self._process_window_prepared_no_padding_with_source(
                win_tok,
                rand_states,
                prev_states,
                prev_state_source,
                chunk_offset_vec,
                positions_tensor,
                chunk_attn_mask,
                alpha,
                collect_gate_stats=collect_gate_stats,
                token_keep_win=token_keep_win,
            )
        return self._process_window_prepared_no_padding_no_source(
            win_tok,
            rand_states,
            prev_states,
            chunk_offset_vec,
            positions_tensor,
            chunk_attn_mask,
            alpha,
            collect_gate_stats=collect_gate_stats,
        )

    def _process_window_prepared_no_padding_no_source(
        self,
        win_tok,
        rand_states,
        prev_states,
        chunk_offset_vec,
        positions_tensor,
        chunk_attn_mask,
        alpha,
        collect_gate_stats: bool = False,
        token_keep_win: Optional[torch.Tensor] = None,
    ):
        """Process a full-validity window with no vertical source tensor."""
        return self._process_window_prepared(
            win_tok,
            rand_states,
            prev_states,
            None,
            chunk_offset_vec,
            positions_tensor,
            chunk_attn_mask,
            None,
            None,
            alpha,
            collect_gate_stats=collect_gate_stats,
            token_keep_win=token_keep_win,
        )

    def _process_window_prepared_no_padding_with_source(
        self,
        win_tok,
        rand_states,
        prev_states,
        prev_state_source,
        chunk_offset_vec,
        positions_tensor,
        chunk_attn_mask,
        alpha,
        collect_gate_stats: bool = False,
        token_keep_win: Optional[torch.Tensor] = None,
    ):
        """Process a full-validity window with a vertical source tensor."""
        return self._process_window_prepared(
            win_tok,
            rand_states,
            prev_states,
            prev_state_source,
            chunk_offset_vec,
            positions_tensor,
            chunk_attn_mask,
            None,
            None,
            alpha,
            collect_gate_stats=collect_gate_stats,
            token_keep_win=token_keep_win,
        )

    def _process_window_prepared_with_padding_no_source(
        self,
        win_tok,
        rand_states,
        prev_states,
        chunk_offset_vec,
        positions_tensor,
        chunk_attn_mask,
        window_pad_indicator,
        padded_rows,
        alpha,
        collect_gate_stats: bool = False,
        token_keep_win: Optional[torch.Tensor] = None,
    ):
        """Process a padded-boundary window with no vertical source tensor."""
        return self._process_window_prepared(
            win_tok,
            rand_states,
            prev_states,
            None,
            chunk_offset_vec,
            positions_tensor,
            chunk_attn_mask,
            window_pad_indicator,
            padded_rows,
            alpha,
            collect_gate_stats=collect_gate_stats,
            token_keep_win=token_keep_win,
        )

    def _process_window_prepared_with_padding_with_source(
        self,
        win_tok,
        rand_states,
        prev_states,
        prev_state_source,
        chunk_offset_vec,
        positions_tensor,
        chunk_attn_mask,
        window_pad_indicator,
        padded_rows,
        alpha,
        collect_gate_stats: bool = False,
        token_keep_win: Optional[torch.Tensor] = None,
    ):
        """Process a padded-boundary window with a vertical source tensor."""
        return self._process_window_prepared(
            win_tok,
            rand_states,
            prev_states,
            prev_state_source,
            chunk_offset_vec,
            positions_tensor,
            chunk_attn_mask,
            window_pad_indicator,
            padded_rows,
            alpha,
            collect_gate_stats=collect_gate_stats,
            token_keep_win=token_keep_win,
        )

    def _process_window_prepared(
        self,
        win_tok,
        rand_states,
        prev_states,
        prev_state_source,
        chunk_offset_vec,
        positions_tensor,
        chunk_attn_mask,
        window_pad_indicator,
        padded_rows,
        alpha,
        collect_gate_stats: bool = False,
        token_keep_win: Optional[torch.Tensor] = None,
    ):
        """Process a prepared single window with state handling.

        ``token_keep_win`` is an optional ``[B, window_size]`` boolean tensor
        (True = keep) that excludes input-masked (``patch_mask``) real-token
        slots from the horizontal state-update gate's token mean, so masked
        positions cannot leak into valid tokens through the recurrent state.
        It does NOT zero those token outputs (unlike window-fill geometry
        padding), so the model still emits a representation at masked
        positions. When None, behaviour is identical to the pre-mask code path.
        """
        B = win_tok.size(0)
        device = win_tok.device

        def norm_states(t: torch.Tensor) -> torch.Tensor:
            return self.state_norm(t)

        window_gate_stats = None
        push_gate_buffer_stats = collect_gate_stats or not self.compile_per_window

        # Handle experiment modes
        if self.experiment_mode == "pmt_stateless":
            current_states = norm_states(rand_states)
            prev_states = self.sos_state_tokens.unsqueeze(0).expand(B, -1, -1).clone()
        elif self.experiment_mode == "pmt_stateful_random":
            current_states = norm_states(rand_states)
        else:  # Normal PMT mode
            if prev_state_source is None:
                current_states = norm_states(rand_states)
            elif self.gate is not None:
                # Apply normalization AFTER gating to preserve signal for learning
                if collect_gate_stats and getattr(self.gate, "supports_collect_stats", False):
                    gated_states = self.gate(prev_state_source, rand_states, collect_stats=True, push_stats=True)
                else:
                    gated_states = self.gate(
                        prev_state_source,
                        rand_states,
                        collect_stats=collect_gate_stats,
                        push_stats=push_gate_buffer_stats,
                    )
                current_states = norm_states(gated_states)
                if collect_gate_stats and hasattr(self.gate, 'last_gate_stats'):
                    window_gate_stats = self.gate.last_gate_stats
            else:
                current_states = (
                    norm_states(alpha * prev_state_source + (1.0 - alpha) * rand_states)
                    if alpha is not None
                    else norm_states(rand_states)
                )
                if collect_gate_stats and alpha is not None:
                    window_gate_stats = {"mean": float(alpha), "std": 0.0, "min": float(alpha), "max": float(alpha)}

        # Track the horizontal carrier entering this window
        prev_states_in = prev_states

        # Apply chunk offset embedding
        if chunk_offset_vec is not None:
            current_states = current_states + chunk_offset_vec
            prev_states_in = prev_states_in + chunk_offset_vec

        # Prepare transformer input
        # Keep recurrent state math in state precision. Under AMP, avoid promoting
        # the entire sequence to fp32 via mixed-dtype concatenation.
        amp_enabled = bool(torch.is_autocast_enabled())
        if not amp_enabled:
            try:
                amp_enabled = bool(torch.is_autocast_enabled(device_type=device.type))
            except TypeError:
                if device.type == "cpu" and hasattr(torch, "is_autocast_cpu_enabled"):
                    amp_enabled = bool(torch.is_autocast_cpu_enabled())

        if amp_enabled and win_tok.dtype in (torch.float16, torch.bfloat16):
            attn_dtype = win_tok.dtype
            prev_states_in_attn = (
                prev_states_in if prev_states_in.dtype == attn_dtype else prev_states_in.to(dtype=attn_dtype)
            )
            current_states_attn = (
                current_states if current_states.dtype == attn_dtype else current_states.to(dtype=attn_dtype)
            )
            input_seq = torch.cat([prev_states_in_attn, current_states_attn, win_tok], dim=1)
        else:
            input_seq = torch.cat([prev_states_in, current_states, win_tok], dim=1)

        # Pass through transformer
        layer_kwargs = dict(
            attn_mask=chunk_attn_mask, positions=positions_tensor, use_rope=self.use_rope, rope_base=self.rope_base
        )
        if getattr(self.layer, "supports_attn_mask_sanitized", False):
            layer_kwargs["attn_mask_sanitized"] = True
        seq_t = input_seq.transpose(0, 1)

        def _run_layer(seq_in_t: torch.Tensor) -> torch.Tensor:
            return self.layer(seq_in_t, **layer_kwargs)

        use_checkpoint = should_use_window_checkpointing(
            window_checkpointing=self.window_checkpointing, training=self.training, grad_enabled=torch.is_grad_enabled()
        )
        out = run_with_optional_window_checkpoint(_run_layer, seq_t, enable_checkpoint=use_checkpoint).transpose(0, 1)

        if self.debug:
            assert torch.isfinite(out).all(), f"{self.name}: Layer produced NaNs in prepared window"
        if padded_rows is not None:
            keep_rows = (~padded_rows).to(out.dtype).view(1, -1, 1)
            out = out * keep_rows

        # Extract updated states
        if self.experiment_mode != "pmt_stateless":
            candidate_states = out[:, self.n_states : 2 * self.n_states]
            # Keep state updates in the state stream precision (typically fp32).
            if candidate_states.dtype != current_states.dtype:
                candidate_states = candidate_states.to(dtype=current_states.dtype)
            candidate_states = norm_states(candidate_states)

        # Keep a pre-injection copy for update-gate features (Phase 1.1).
        # This avoids feeding read-injected state signals back into update inputs.
        window_out_raw = out[:, 2 * self.n_states :]
        window_out = window_out_raw

        # State -> token read path (Phase 1.2) using carry state only
        if (
            self.experiment_mode != "pmt_stateless"
            and self.state_read_gate is not None
            and self.state_read_proj is not None
            and self.n_states > 0
        ):
            prev_states_in_norm = norm_states(prev_states_in)
            read_src = prev_states_in_norm.mean(dim=1)
            gate_logits = self.state_read_gate(read_src) / self.state_read_temperature
            gate_vals = torch.sigmoid(gate_logits)
            if gate_vals.shape[-1] == 1:
                gate_vals = gate_vals.view(B, 1, 1)
            else:
                gate_vals = gate_vals.view(B, 1, -1)
            read_proj = self.state_read_proj(read_src).unsqueeze(1)
            window_out = window_out + gate_vals * read_proj
            if push_gate_buffer_stats:
                push_gate_stats(f"{self.name}_state_read_gate", gate_vals.squeeze(1))
            if window_pad_indicator is not None:
                # Keep padded token rows zero after read injection without bool-index sync.
                keep_tokens = window_pad_indicator.to(window_out.dtype).view(1, -1, 1)
                window_out = window_out * keep_tokens

        # Horizontal update gate (Phase 1.1)
        if self.experiment_mode != "pmt_stateless" and self.state_update_gate is not None and self.n_states > 0:
            prev_states_in_norm = norm_states(prev_states_in)
            if token_keep_win is not None:
                # Exclude input-masked (patch_mask) tokens AND geometry-fill
                # tokens from the carried-state token mean so masked positions
                # cannot leak into valid tokens through the horizontal state.
                keep = token_keep_win.to(window_out_raw.dtype)  # [B, window_size]
                if window_pad_indicator is not None:
                    keep = keep * window_pad_indicator.to(window_out_raw.dtype).unsqueeze(0)
                token_mask = keep.unsqueeze(-1)  # [B, window_size, 1]
                tok_sum = (window_out_raw * token_mask).sum(dim=1)
                tok_den = token_mask.sum(dim=1).clamp_min(1.0)
                tok_mean = tok_sum / tok_den
            elif window_pad_indicator is not None:
                token_mask = window_pad_indicator.to(window_out_raw.dtype).unsqueeze(0).unsqueeze(-1)
                tok_sum = (window_out_raw * token_mask).sum(dim=1)
                tok_den = token_mask.sum(dim=1).clamp_min(1.0)
                tok_mean = tok_sum / tok_den
            else:
                tok_mean = window_out_raw.mean(dim=1)
            prev_mean = prev_states_in_norm.mean(dim=1)
            cand_mean = candidate_states.mean(dim=1)
            gate_in = torch.cat([prev_mean, cand_mean, tok_mean], dim=-1)
            gate_logits = self.state_update_gate(gate_in) / self.state_update_gate_temperature
            gate_vals = torch.sigmoid(gate_logits)
            if gate_vals.shape[-1] == 1:
                gate_vals = gate_vals.view(B, 1, 1)
            else:
                gate_vals = gate_vals.view(B, 1, -1)
            updated_states = prev_states_in_norm + gate_vals * (candidate_states - prev_states_in_norm)
            prev_states = norm_states(updated_states)
            if push_gate_buffer_stats:
                push_gate_stats(f"{self.name}_state_update_gate", gate_vals.squeeze(1))
        elif self.experiment_mode != "pmt_stateless":
            prev_states = candidate_states

        return window_out, prev_states, window_gate_stats

    def _process_window(
        self,
        win_tok,
        w_i,
        rand_states,
        prev_states,
        prev_block_states,
        prev_block_all_states,
        alpha,
        front_pad,
        seq_len,
        absolute_positions,
        device,
        collect_gate_stats: bool = False,
        positions_for_windows: Optional[torch.Tensor] = None,
        patch_invalid_win: Optional[torch.Tensor] = None,
    ):
        """Process a single window with state handling.

        ``patch_invalid_win`` is an optional ``[B, window_size]`` boolean tensor
        (True = invalid under ``patch_mask``). It drives both the chunk
        attention mask (so invalid tokens are not attended to) and the
        state-gate token-mean exclusion (so they do not leak through the
        horizontal state carrier).
        """
        prepared = self._prepare_window_runtime(
            w_i=int(w_i),
            prev_block_states=prev_block_states,
            prev_block_all_states=prev_block_all_states,
            front_pad=int(front_pad),
            seq_len=int(seq_len),
            positions_for_windows=positions_for_windows,
            absolute_positions=absolute_positions,
            device=device,
            patch_invalid_win=patch_invalid_win,
        )
        # Only build the gate token-keep vector when this window actually hides
        # real tokens; otherwise keep the byte-identical no-op gate path.
        token_keep_win = None
        if patch_invalid_win is not None and bool(patch_invalid_win.any()):
            token_keep_win = ~patch_invalid_win.to(device=device, dtype=torch.bool)
        return self._run_process_window_prepared(
            win_tok,
            rand_states,
            prev_states,
            *prepared,
            alpha,
            collect_gate_stats=collect_gate_stats,
            token_keep_win=token_keep_win,
        )

    def forward_windows(
        self,
        x: torch.Tensor,
        prev_block_states: Optional[torch.Tensor] = None,
        prev_block_all_states: Optional[torch.Tensor] = None,
        prev_window_offset: Optional[int] = None,
        window_offset: int = 0,
        alpha: Optional[float] = None,
        patch_mask: Optional[torch.BoolTensor] = None,
        return_gate_stats: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.BoolTensor], Dict]:
        """Process input and return per-window embeddings for token stitching.

        Args:
            x: Input tensor [B, L, D]
            prev_block_states: Previous block's final states [B, n_states, D]
            prev_block_all_states: Previous block's all window states [B, num_win, n_states, D]
            prev_window_offset: Previous block offset used to build prev_block_all_states
            window_offset: Current block window offset
            alpha: Mixing coefficient for state reuse
            patch_mask: Valid token mask [B, L]
            return_gate_stats: Whether to return gate statistics

        Returns:
            Tuple of:
                - window_emb: Window embeddings [B, num_win, W, D]
                - final_states: Final state tokens [B, n_states, D]
                - all_states: All window states [B, num_win, n_states, D]
                - patch_mask: Updated valid token mask [B, L]
                - meta: Geometry metadata dict
        """
        prep = self._prepare_windows(x, patch_mask, window_offset=window_offset)
        B = x.size(0)
        patch_invalid_windows = prep.get('patch_invalid_windows')

        if self.debug and patch_invalid_windows is not None and not getattr(self, "_warned_patch_mask_invalid", False):
            # patch_mask invalid tokens are now masked out of chunk attention and
            # excluded from the horizontal state-update gate. Emit an info note
            # once so the behaviour change is visible in debug runs.
            logger.info(
                "%s: applying patch_mask to chunk attention (%d invalid window-slots); "
                "invalid tokens are blocked as attention keys/queries and excluded from state gating.",
                self.name,
                int(patch_invalid_windows.sum().item()),
            )
            self._warned_patch_mask_invalid = True

        sos_prefix = self.sos_state_tokens.unsqueeze(0).expand(B, -1, -1).contiguous()
        base_states = self.base_state.unsqueeze(0).expand(B, -1, -1).contiguous()
        window_emb = prep['windows'].new_empty(B, prep['num_win'], self.window_size, self.d_model)
        all_states = sos_prefix.new_empty(B, prep['num_win'], self.n_states, self.d_model)
        prev_states = sos_prefix
        carry_states = sos_prefix
        routed_prev_block_all_states = prev_block_all_states
        if prev_block_all_states is not None:
            if prev_window_offset is None:
                src_offset = 0
                if self.debug and not getattr(self, "_warned_missing_prev_window_offset", False):
                    logger.warning(
                        "%s: prev_window_offset missing for routed vertical states; assuming previous offset=0.",
                        self.name,
                    )
                    self._warned_missing_prev_window_offset = True
            else:
                src_offset = int(prev_window_offset)
            routed_prev_block_all_states = self._route_window_states(
                prev_block_all_states,
                prev_offset=src_offset,
                cur_offset=int(window_offset),
                num_win_cur=prep['num_win'],
            )

        # Process each window
        for w_i in range(prep['num_win']):
            win_tok = prep['windows'][:, w_i]
            rand_states = base_states
            if self.training and self.state_init_noise_std > 0.0:
                rand_states = rand_states + (
                    self.state_init_noise_std
                    * torch.randn(B, self.n_states, self.d_model, device=prep['device'], dtype=base_states.dtype)
                )

            states_in = carry_states if self.use_horizontal_state_passing else sos_prefix
            patch_invalid_win = None if patch_invalid_windows is None else patch_invalid_windows[:, w_i]
            window_out, prev_states, window_gate_stats = self._process_window(
                win_tok,
                w_i,
                rand_states,
                states_in,
                prev_block_states,
                routed_prev_block_all_states,
                alpha,
                prep['front_pad'],
                prep['L'],
                prep['absolute_positions'],
                prep['device'],
                collect_gate_stats=return_gate_stats,
                positions_for_windows=prep.get('positions_for_windows'),
                patch_invalid_win=patch_invalid_win,
            )

            window_emb[:, w_i] = window_out
            all_states[:, w_i] = prev_states

            if self.use_horizontal_state_passing:
                carry_states = prev_states
            else:
                carry_states = sos_prefix

        # Keep final state semantics consistent with the recurrent path.
        final_states = all_states[:, -1]

        # Update mask if needed
        valid_len = prep['L']
        if patch_mask is not None:
            valid_len = min(prep['L'], prep['Lp'] - prep['front_pad'])
            if valid_len < prep['L']:
                patch_mask = patch_mask[:, :valid_len]

        meta = {
            "front_pad": prep['front_pad'],
            "Lp": prep['Lp'],
            "L": prep['L'],
            "window_size": self.window_size,
            "window_stride": self.window_stride,
            "num_win": prep['num_win'],
            "window_offset": int(window_offset),
        }

        return window_emb, final_states, all_states, patch_mask, meta

    def forward(
        self,
        x: torch.Tensor,
        prev_block_states: Optional[torch.Tensor] = None,
        prev_block_all_states: Optional[torch.Tensor] = None,
        prev_window_offset: Optional[int] = None,
        window_offset: int = 0,
        alpha: Optional[float] = None,
        patch_mask: Optional[torch.BoolTensor] = None,
        return_gate_stats: bool = False,
    ) -> tuple:
        """Process input and return token stream [B, L, D] for sPMA."""
        window_emb, final_states, all_states, patch_mask, meta = self.forward_windows(
            x,
            prev_block_states=prev_block_states,
            prev_block_all_states=prev_block_all_states,
            prev_window_offset=prev_window_offset,
            window_offset=window_offset,
            alpha=alpha,
            patch_mask=patch_mask,
            return_gate_stats=return_gate_stats,
        )

        B = x.size(0)
        L = meta['L']
        # Disjoint windows can be stitched via reshape, then cropped back.
        tokens_padded = window_emb.reshape(B, meta['Lp'], self.d_model)
        tokens_out = tokens_padded[:, meta['front_pad'] : meta['front_pad'] + L, :].contiguous()

        # Handle length alignment (pad/trim to target_len derived from mask or input)
        target_len = max(L, patch_mask.size(1)) if patch_mask is not None else L
        valid_len = tokens_out.size(1)
        if patch_mask is not None and patch_mask.size(1) < target_len:
            # Pad mask with valid tokens to match target length
            pad_len = target_len - patch_mask.size(1)
            patch_mask = torch.cat(
                [patch_mask, torch.ones(patch_mask.size(0), pad_len, device=patch_mask.device, dtype=patch_mask.dtype)],
                dim=1,
            )
        if valid_len != target_len:
            if valid_len < target_len:
                if not getattr(self, "_warned_len_pad", False):
                    logger.warning(
                        f"{self.name}: Token length {valid_len} shorter than target {target_len}; padding with zeros."
                    )
                    self._warned_len_pad = True
                pad_len = target_len - valid_len
                pad = torch.zeros(
                    tokens_out.size(0), pad_len, self.d_model, device=tokens_out.device, dtype=tokens_out.dtype
                )
                tokens_out = torch.cat([tokens_out, pad], dim=1)
            else:
                if not getattr(self, "_warned_len_trim", False):
                    logger.warning(
                        f"{self.name}: Token length {valid_len} exceeds target {target_len}; trimming output."
                    )
                    self._warned_len_trim = True
                tokens_out = tokens_out[:, :target_len]
            if patch_mask is not None and patch_mask.size(1) != target_len:
                patch_mask = patch_mask[:, :target_len]

        # Process gate statistics
        aggregated_stats = {}
        if return_gate_stats and hasattr(self, 'gate') and self.gate is not None:
            # Aggregate statistics from windows
            if hasattr(self.gate, 'last_gate_stats') and self.gate.last_gate_stats:
                aggregated_stats = self.gate.last_gate_stats

        if return_gate_stats:
            return tokens_out, final_states, all_states, patch_mask, aggregated_stats
        else:
            return tokens_out, final_states, all_states, patch_mask


# -------------------------------------------------------------------
#  SkipGate - Learnable residual connection gating
# -------------------------------------------------------------------
# SkipGate is now imported from trd.hosts.pmt.components


# -------------------------------------------------------------------
#  PMABlock - Single transformer block with PMA components
# -------------------------------------------------------------------
class PMABlock(nn.Module):
    """Single PMA transformer block.

    Combines chunk processing, overlap aggregation, and MLP with
    learnable skip connections. Supports both pre-norm and post-norm
    architectures.

    Architecture:
        Pre-norm:  x → LN → Chunk → Agg → Skip → LN → MLP → Out
        Post-norm: x → Chunk → Agg → Skip → LN → MLP → LN → Out

    Args:
        d_model: Hidden dimension
        num_heads: Number of attention heads
        d_ff: Feedforward dimension
        n_states: Number of state tokens
        window_size: Window size for chunking
        window_stride: Stride between windows
        max_O: Maximum overlap size
        norm_type: Normalization type ('ln' or 'rms')
        norm_position: 'pre' or 'post'
        drop_block_norm: Whether to drop redundant norms
        cross_block_state: Whether to reuse states from previous block
        gate_type: Type of state mixing gate
        stream_chunk_len: Tile size for memory-bounded processing in standard overlap aggregation
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        n_states: int,
        window_size: int,
        window_stride: int,
        max_O: int,
        norm_type="ln",
        norm_position="pre",
        reduce_norms=False,  # Deprecated
        drop_block_norm=True,
        mlp_pre_norm=False,
        activation='relu',
        dropout=0.1,
        masking='causal',
        use_rope=False,
        rope_base=10000.0,
        local_offset_scale=0.1,
        use_chunk_offset=False,
        rope_chunk_offset_on_states: bool = False,
        max_chunks=10000,
        chunk_offset_scale=1.0,
        debug=False,
        name="PMABlock",
        cross_block_state=False,
        cross_block_alpha=0.5,
        gate_type=None,
        cross_block_all_states=False,
        pma_aggregator_use_conv_q=True,
        skip_gate_kind: str = "dimwise",
        skip_gate_temperature: float = 1.0,
        skip_gate_init_logit: float = -2.0,
        residual_init: float = 0.0,
        dropout_mode="legacy",
        learnable_gamma: bool = True,
        experiment_mode='pmt',
        stream_chunk_len: Optional[int] = None,
        overlap_mode: str = 'attn',
        shift_pattern: str = "alternate",
        gate_temperature: float = 0.5,
        gate_use_residual: bool = True,
        gate_init_logit: float = None,
        use_horizontal_state_passing: bool = True,
        allow_state_to_attend_prev_state: bool = False,
        # Optional: bypass skip-gate in selector mode (parity with v6 recommendation)
        disable_selector_skip_gate: bool = True,
        state_init_noise_std: float = 0.01,
        window_checkpointing: bool = False,
        state_update_gate_enabled: bool = False,
        state_update_gate_kind: str = "dimwise",
        state_update_gate_init_logit: float = -2.0,
        state_update_gate_temperature: float = 1.0,
        state_read_enabled: bool = False,
        state_read_kind: str = "dimwise",
        state_read_init_logit: float = -2.0,
        state_read_temperature: float = 1.0,
        compile_per_window: bool = False,
        compile_mode: str = "reduce-overhead",
    ):
        super().__init__()
        self.name = name
        self.norm_type = norm_type
        self.experiment_mode = experiment_mode
        self.norm_position = norm_position.lower()
        self.drop_block_norm = drop_block_norm
        self.mlp_pre_norm = mlp_pre_norm
        self.cross_block_state = cross_block_state
        self.cross_block_alpha = cross_block_alpha
        self.gate_type = gate_type
        self.cross_block_all_states = cross_block_all_states
        self.debug = debug
        self.overlap_mode = overlap_mode
        self.disable_selector_skip_gate = bool(disable_selector_skip_gate)
        self.use_horizontal_state_passing = use_horizontal_state_passing
        self.allow_state_to_attend_prev_state = allow_state_to_attend_prev_state
        if self.overlap_mode != "attn":
            raise ValueError(
                f"{self.name}: sPMA v5.4 only supports overlap_mode='attn' (legacy overlap resolvers are removed)."
            )

        if reduce_norms:
            import warnings

            warnings.warn("reduce_norms is deprecated; use drop_block_norm instead", DeprecationWarning, stacklevel=2)

        # Setup normalization layers
        use_identity_norm1 = drop_block_norm and norm_position == "pre"
        use_identity_norm2 = drop_block_norm and not mlp_pre_norm and norm_position == "pre"

        self.norm1 = nn.Identity() if use_identity_norm1 else choose_norm(norm_type, d_model)
        self.norm2 = nn.Identity() if use_identity_norm2 else choose_norm(norm_type, d_model)
        self.norm3 = nn.Identity() if drop_block_norm else choose_norm(norm_type, d_model)

        # Core components
        self.chunk_processor = PMAChunkProcessor(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            n_states=n_states,
            window_size=window_size,
            window_stride=window_stride,
            activation=activation,
            dropout=dropout,
            masking=masking,
            use_rope=use_rope,
            rope_base=rope_base,
            local_offset_scale=local_offset_scale,
            use_chunk_offset=use_chunk_offset,
            rope_chunk_offset_on_states=rope_chunk_offset_on_states,
            max_chunks=max_chunks,
            chunk_offset_scale=chunk_offset_scale,
            experiment_mode=experiment_mode,
            norm_type=norm_type,
            norm_position=norm_position,
            debug=debug,
            name=f"{name}_ChunkProcessorVec",
            gate_type=gate_type if cross_block_state else None,
            dropout_mode=dropout_mode,
            gate_temperature=gate_temperature,
            gate_use_residual=gate_use_residual,
            gate_init_logit=gate_init_logit,
            use_horizontal_state_passing=use_horizontal_state_passing,
            allow_state_to_attend_prev_state=allow_state_to_attend_prev_state,
            state_init_noise_std=state_init_noise_std,
            window_checkpointing=window_checkpointing,
            state_update_gate_enabled=state_update_gate_enabled,
            state_update_gate_kind=state_update_gate_kind,
            state_update_gate_init_logit=state_update_gate_init_logit,
            state_update_gate_temperature=state_update_gate_temperature,
            state_read_enabled=state_read_enabled,
            state_read_kind=state_read_kind,
            state_read_init_logit=state_read_init_logit,
            state_read_temperature=state_read_temperature,
            compile_per_window=compile_per_window,
            compile_mode=compile_mode,
        )

        self.selector = None
        self.aggregator = None

        self.skip_gate = SkipGate(d_model, skip_gate_kind, temp=skip_gate_temperature, init_logit=skip_gate_init_logit)
        # Normalize skip branch inputs before gating for symmetric scaling
        self.resid_ln = choose_norm(norm_type, d_model)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation.lower() == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x,
        *,
        patch_mask: Optional[torch.BoolTensor] = None,
        prev_block_states=None,
        prev_block_all_states=None,
        prev_window_offset: Optional[int] = None,
        window_offset: int = 0,
        return_gate_stats: bool = False,
    ):
        """Forward pass through the block."""
        # Ensure patch_mask is at least as long as the input to avoid repeated trimming/padding downstream
        if patch_mask is not None and patch_mask.size(1) < x.size(1):
            pad_len = x.size(1) - patch_mask.size(1)
            patch_mask = torch.cat(
                [patch_mask, torch.ones(patch_mask.size(0), pad_len, device=patch_mask.device, dtype=patch_mask.dtype)],
                dim=1,
            )
            if not getattr(self, "_warned_patch_mask_pad", False):
                logger.warning(
                    f"{self.name}: patch_mask shorter than input (mask={patch_mask.size(1)-pad_len}, x={x.size(1)}); padding mask to match."
                )
                self._warned_patch_mask_pad = True

        # Log cross-block state usage
        # Skip verbose cross-block state logging to reduce noise in debug mode

        # Determine alpha for static mixing
        current_alpha = self.cross_block_alpha if self.cross_block_state and self.gate_type is None else None

        # Select appropriate previous states
        if self.cross_block_state and self.cross_block_all_states and prev_block_all_states is not None:
            prev_states_to_use = (None, prev_block_all_states)
        else:
            prev_states_to_use = (prev_block_states if self.cross_block_state else None, None)

        # Apply architecture-specific processing
        if self.norm_position == "pre":
            x_in = self.norm1(x)
        else:
            x_in = x

        # sPMA token-stream path [B, L, D]
        if return_gate_stats:
            token_out, final_states, all_states, patch_mask, gate_stats = self.chunk_processor(
                x_in,
                prev_block_states=prev_states_to_use[0],
                prev_block_all_states=prev_states_to_use[1],
                prev_window_offset=prev_window_offset,
                window_offset=window_offset,
                alpha=current_alpha,
                patch_mask=patch_mask,
                return_gate_stats=True,
            )
        else:
            token_out, final_states, all_states, patch_mask = self.chunk_processor(
                x_in,
                prev_block_states=prev_states_to_use[0],
                prev_block_all_states=prev_states_to_use[1],
                prev_window_offset=prev_window_offset,
                window_offset=window_offset,
                alpha=current_alpha,
                patch_mask=patch_mask,
                return_gate_stats=False,
            )
            gate_stats = {}

        # Check dimension consistency
        if patch_mask is not None and patch_mask.size(1) != token_out.size(1):
            min_L = min(patch_mask.size(1), token_out.size(1))
            logger.warning(f"{self.name}: Dimension mismatch - trimming to {min_L}")
            token_out = token_out[:, :min_L]
            patch_mask = patch_mask[:, :min_L]

        # Align all paths before residual update.
        candidate_lens = [token_out.size(1), x.size(1), x_in.size(1)]
        if patch_mask is not None:
            candidate_lens.append(patch_mask.size(1))
        target_len = min(candidate_lens)
        if target_len <= 0:
            raise ValueError(f"{self.name}: Non-positive target sequence length after alignment.")
        if any(length != target_len for length in candidate_lens):
            if not getattr(self, "_warned_residual_trim", False):
                logger.warning(
                    f"{self.name}: Residual length mismatch (x={x.size(1)}, x_in={x_in.size(1)}, "
                    f"token_out={token_out.size(1)}, mask={patch_mask.size(1) if patch_mask is not None else 'None'}); "
                    f"trimming to {target_len}"
                )
                self._warned_residual_trim = True
            token_out = token_out[:, :target_len]
            x = x[:, :target_len]
            x_in = x_in[:, :target_len]
            if patch_mask is not None and patch_mask.size(1) != target_len:
                patch_mask = patch_mask[:, :target_len]

        # Use a true residual update. token_out includes internal residuals from x_in.
        # Subtract x_in so the outer residual path receives only the learned update.
        token_update = token_out - x_in
        # Keep SkipGate input non-zero so tokenwise gates remain learnable.
        # Since SkipGate returns `skip + gate * agg`, subtract skip afterward.
        skip_for_gate = self.resid_ln(x_in)
        if return_gate_stats and getattr(self.skip_gate, "supports_collect_stats", False):
            gated_with_skip = self.skip_gate(skip_for_gate, token_update, collect_stats=True)
        else:
            gated_with_skip = self.skip_gate(skip_for_gate, token_update, collect_stats=False)
        fused = gated_with_skip - skip_for_gate

        # Add outer residual connection
        x_out = x + fused

        # Collect skip gate statistics
        if return_gate_stats and hasattr(self.skip_gate, 'last_gate_stats'):
            if 'skip_gate' not in gate_stats:
                gate_stats['skip_gate'] = {}
            gate_stats['skip_gate'].update(self.skip_gate.last_gate_stats)

        # Apply remaining layers based on norm position
        if self.norm_position == "pre":
            x_out = self.norm2(x_out)
            mlp_out = self.mlp(x_out)
            x_out = x_out + mlp_out
            x_out = self.norm3(x_out)
        else:
            x_out = self.norm1(x_out)
            mlp_out = self.mlp(x_out)
            x_out = x_out + mlp_out
            x_out = self.norm2(x_out)

        if return_gate_stats:
            return x_out, final_states, all_states, patch_mask, gate_stats
        else:
            return x_out, final_states, all_states, patch_mask


# -------------------------------------------------------------------
#  StackedPMA - Sequential composition of PMA blocks
# -------------------------------------------------------------------
class StackedPMA(nn.Module):
    """Stack of PMA blocks with optional cross-block state reuse.

    Composes multiple PMABlocks sequentially with configurable
    state passing between blocks.

    Args:
        num_blocks: Number of PMA blocks
        cross_block_state: Whether to pass states between blocks
        cross_block_all_states: Whether to pass all window states
        gate_type: Type of state mixing gate
        apply_terminal_norm: Keep the legacy final-block export normalization
        Other args passed to PMABlock constructors
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        n_states: int,
        window_size: int = None,
        window_stride: int = None,
        norm_type="ln",
        norm_position="pre",
        reduce_norms=False,
        drop_block_norm=True,
        mlp_pre_norm=False,
        activation='relu',
        dropout=0.1,
        masking='causal',
        use_rope=False,
        rope_base=10000.0,
        local_offset_scale=0.1,
        use_chunk_offset=False,
        rope_chunk_offset_on_states: bool = False,
        max_chunks=10000,
        chunk_offset_scale=1.0,
        debug=False,
        cross_block_state=False,
        cross_block_alpha=0.5,
        gate_type=None,
        cross_block_all_states=False,
        pma_aggregator_use_conv_q=True,
        skip_gate_kind: str = "dimwise",
        skip_gate_temperature: float = 1.0,
        skip_gate_init_logit: float = -2.0,
        residual_init: float = 0.0,
        dropout_mode="legacy",
        learnable_gamma: bool = True,
        experiment_mode='pmt',
        stream_chunk_len: Optional[int] = None,
        overlap_mode: str = 'attn',
        shift_pattern: str = "alternate",
        gate_temperature: float = 0.5,
        gate_use_residual: bool = True,
        gate_init_logit: float = None,
        use_horizontal_state_passing: bool = True,
        allow_state_to_attend_prev_state: bool = False,
        state_init_noise_std: float = 0.01,
        window_checkpointing: bool = False,
        state_update_gate_enabled: bool = False,
        state_update_gate_kind: str = "dimwise",
        state_update_gate_init_logit: float = -2.0,
        state_update_gate_temperature: float = 1.0,
        state_read_enabled: bool = False,
        state_read_kind: str = "dimwise",
        state_read_init_logit: float = -2.0,
        state_read_temperature: float = 1.0,
        compile_per_window: bool = False,
        compile_mode: str = "reduce-overhead",
        apply_terminal_norm: bool = True,
    ):
        super().__init__()
        self.cross_block_state = cross_block_state
        self.cross_block_alpha = cross_block_alpha
        self.stream_chunk_len = stream_chunk_len
        self.gate_type = gate_type
        self.cross_block_all_states = cross_block_all_states
        self.debug = debug
        self.experiment_mode = experiment_mode
        self.overlap_mode = overlap_mode
        self.shift_pattern = str(shift_pattern).lower()
        self.use_horizontal_state_passing = use_horizontal_state_passing
        self.allow_state_to_attend_prev_state = allow_state_to_attend_prev_state
        self.window_checkpointing = bool(window_checkpointing)
        self.compile_per_window = bool(compile_per_window)
        self.compile_mode = str(compile_mode or "reduce-overhead")
        self.apply_terminal_norm = bool(apply_terminal_norm)
        if self.shift_pattern not in {"alternate", "pairs"}:
            raise ValueError(f"Unsupported shift_pattern '{shift_pattern}'. Use 'alternate' or 'pairs'.")

        if reduce_norms:
            import warnings

            warnings.warn("reduce_norms is deprecated; use drop_block_norm instead", DeprecationWarning, stacklevel=2)

        # Handle None values for window_size and window_stride
        # Use reasonable defaults if not provided
        if window_size is None:
            window_size = 8  # Default window size
        if window_stride is None:
            window_stride = window_size
        if window_stride != window_size:
            if debug:
                logger.info(
                    "StackedPMA(v5.4): overriding window_stride=%s to window_size=%s for non-overlapping windows.",
                    window_stride,
                    window_size,
                )
            window_stride = window_size

        self.window_size = int(window_size)
        self.window_stride = int(window_stride)
        self.shift_size = self.window_size // 2

        # Calculate maximum overlap
        auto_max_O = 1

        # Build blocks
        blocks = []
        for i in range(num_blocks):
            block = PMABlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                n_states=n_states,
                window_size=window_size,
                window_stride=window_stride,
                max_O=auto_max_O,
                norm_type=norm_type,
                norm_position=norm_position,
                reduce_norms=reduce_norms,
                drop_block_norm=drop_block_norm,
                mlp_pre_norm=mlp_pre_norm,
                activation=activation,
                dropout=dropout,
                masking=masking,
                use_rope=use_rope,
                rope_base=rope_base,
                local_offset_scale=local_offset_scale,
                use_chunk_offset=use_chunk_offset,
                rope_chunk_offset_on_states=rope_chunk_offset_on_states,
                max_chunks=max_chunks,
                chunk_offset_scale=chunk_offset_scale,
                debug=debug,
                name=f"Block{i+1}",
                cross_block_state=cross_block_state,
                cross_block_alpha=cross_block_alpha,
                gate_type=gate_type,
                cross_block_all_states=cross_block_all_states,
                pma_aggregator_use_conv_q=pma_aggregator_use_conv_q,
                skip_gate_kind=skip_gate_kind,
                skip_gate_temperature=skip_gate_temperature,
                skip_gate_init_logit=skip_gate_init_logit,
                residual_init=residual_init,
                dropout_mode=dropout_mode,
                learnable_gamma=learnable_gamma,
                experiment_mode=experiment_mode,
                stream_chunk_len=stream_chunk_len,
                overlap_mode=overlap_mode,
                gate_temperature=gate_temperature,
                gate_use_residual=gate_use_residual,
                gate_init_logit=gate_init_logit,
                use_horizontal_state_passing=use_horizontal_state_passing,
                allow_state_to_attend_prev_state=allow_state_to_attend_prev_state,
                state_init_noise_std=state_init_noise_std,
                window_checkpointing=window_checkpointing,
                state_update_gate_enabled=state_update_gate_enabled,
                state_update_gate_kind=state_update_gate_kind,
                state_update_gate_init_logit=state_update_gate_init_logit,
                state_update_gate_temperature=state_update_gate_temperature,
                state_read_enabled=state_read_enabled,
                state_read_kind=state_read_kind,
                state_read_init_logit=state_read_init_logit,
                state_read_temperature=state_read_temperature,
                compile_per_window=compile_per_window,
                compile_mode=compile_mode,
            )

            # Name gates for diagnostics
            block.skip_gate.name = f"residual_stream_gate_{i+1}"
            if hasattr(block, 'aggregator') and block.aggregator is not None:
                block.aggregator.name = f"overlap_gate_{i+1}"
            if block.chunk_processor.gate is not None:
                # Standardize naming with v6: 'cross_block_gate_{i}'
                block.chunk_processor.gate.name = f"cross_block_gate_{i+1}"

            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        # Preserve the legacy stack export norm by default. The opt-out is an
        # isolated ablation: internal block norms retain their configured
        # ``drop_block_norm`` behavior and only the final export norm changes.
        if num_blocks > 0:
            if self.apply_terminal_norm and isinstance(self.blocks[-1].norm3, nn.Identity):
                self.blocks[-1].norm3 = choose_norm(norm_type, d_model)
            elif not self.apply_terminal_norm:
                self.blocks[-1].norm3 = nn.Identity()

    def _block_window_offset(self, block_index: int) -> int:
        """Return the window offset for a given block index."""
        if self.shift_size <= 0:
            return 0
        if self.shift_pattern == "alternate":
            return 0 if (block_index % 2 == 0) else self.shift_size
        # "pairs": 0,0,shift,shift,...
        pair_id = (block_index // 2) % 2
        return 0 if pair_id == 0 else self.shift_size

    def forward(self, x, *, patch_mask: Optional[torch.BoolTensor] = None, return_gate_stats: bool = False):
        """Forward pass through all blocks."""
        final_states, all_states = None, None
        prev_block_states = None
        prev_block_all_states = None
        prev_window_offset = None
        gate_stats = [] if return_gate_stats else None

        for i, block in enumerate(self.blocks):
            current_window_offset = self._block_window_offset(i)
            if self.debug and not getattr(self, "_logged_block_state_once", False):
                logger.debug(f"Block {i+1}: cross_block_state={self.cross_block_state}")
                if prev_block_states is not None:
                    logger.debug(f"  prev_block_states shape: {prev_block_states.shape}")
                if prev_block_all_states is not None:
                    logger.debug(f"  prev_block_all_states shape: {prev_block_all_states.shape}")
                logger.debug(
                    "  window_offset=%s (shift_pattern=%s, shift_size=%s)",
                    current_window_offset,
                    self.shift_pattern,
                    self.shift_size,
                )
                self._logged_block_state_once = True

            # Run block with appropriate state passing
            if self.cross_block_state:
                if self.cross_block_all_states:
                    out = block(
                        x,
                        patch_mask=patch_mask,
                        prev_block_states=None,
                        prev_block_all_states=prev_block_all_states,
                        prev_window_offset=prev_window_offset,
                        window_offset=current_window_offset,
                        return_gate_stats=return_gate_stats,
                    )
                else:
                    out = block(
                        x,
                        patch_mask=patch_mask,
                        prev_block_states=prev_block_states,
                        window_offset=current_window_offset,
                        return_gate_stats=return_gate_stats,
                    )
            else:
                out = block(
                    x, patch_mask=patch_mask, window_offset=current_window_offset, return_gate_stats=return_gate_stats
                )

            # Unpack results
            if return_gate_stats:
                x, final_states, all_states, patch_mask, block_gate_stats = out
                if gate_stats is None:
                    gate_stats = []
                gate_stats.append(block_gate_stats)
            else:
                x, final_states, all_states, patch_mask = out

            # Update states for next block
            if self.cross_block_state:
                if self.cross_block_all_states:
                    prev_block_all_states = all_states
                    prev_block_states = None
                    prev_window_offset = current_window_offset
                else:
                    prev_block_states = final_states
                    prev_block_all_states = None
                    prev_window_offset = None
            else:
                prev_block_states = None
                prev_block_all_states = None
                prev_window_offset = None

        # Aggregate gate statistics
        if return_gate_stats:
            if gate_stats and isinstance(gate_stats[0], dict) and "avg" in gate_stats[0]:
                keys = gate_stats[0]["avg"].keys()
                avg_stats = {}
                for k in keys:
                    values = [float(g["avg"][k]) for g in gate_stats]
                    avg_stats[k] = sum(values) / len(values) if values else 0.0
            else:
                avg_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
                if gate_stats:
                    logger.warning(f"Unexpected gate_stats structure: {type(gate_stats[0])}")

            gate_summary = {"per_block": gate_stats, "avg": avg_stats}
            return x, final_states, all_states, patch_mask, gate_summary
        else:
            return x, final_states, all_states, patch_mask, {}

    def forward_with_capture(
        self,
        x,
        capture_blocks: Optional[Tuple[int]] = None,
        *,
        patch_mask: Optional[torch.BoolTensor] = None,
        return_gate_stats: bool = False,
    ):
        """Forward pass with intermediate state capture for visualization."""
        captured = []
        gate_stats = [] if return_gate_stats else None
        prev_block_states = prev_block_all_states = None
        prev_window_offset = None

        for i, block in enumerate(self.blocks):
            current_window_offset = self._block_window_offset(i)
            if not self.cross_block_all_states:
                if return_gate_stats:
                    x, final_states, all_states, patch_mask, block_gate_stats = block(
                        x,
                        patch_mask=patch_mask,
                        prev_block_states=prev_block_states,
                        window_offset=current_window_offset,
                        return_gate_stats=True,
                    )
                    if gate_stats is None:
                        gate_stats = []
                    gate_stats.append(block_gate_stats)
                else:
                    x, final_states, all_states, patch_mask = block(
                        x,
                        patch_mask=patch_mask,
                        prev_block_states=prev_block_states,
                        window_offset=current_window_offset,
                        return_gate_stats=False,
                    )
            else:
                if return_gate_stats:
                    x, final_states, all_states, patch_mask, block_gate_stats = block(
                        x,
                        patch_mask=patch_mask,
                        prev_block_all_states=prev_block_all_states,
                        prev_window_offset=prev_window_offset,
                        window_offset=current_window_offset,
                        return_gate_stats=True,
                    )
                    if gate_stats is None:
                        gate_stats = []
                    gate_stats.append(block_gate_stats)
                else:
                    x, final_states, all_states, patch_mask = block(
                        x,
                        patch_mask=patch_mask,
                        prev_block_all_states=prev_block_all_states,
                        prev_window_offset=prev_window_offset,
                        window_offset=current_window_offset,
                        return_gate_stats=False,
                    )

            # Capture requested blocks
            if capture_blocks and i in capture_blocks:
                captured.append(all_states)  # no detach for gradient flow

            # Update states
            prev_block_states, prev_block_all_states = (
                (final_states, None) if not self.cross_block_all_states else (None, all_states)
            )
            prev_window_offset = current_window_offset if self.cross_block_all_states else None

        # Aggregate statistics
        if return_gate_stats:
            if gate_stats and isinstance(gate_stats[0], dict) and "avg" in gate_stats[0]:
                keys = gate_stats[0]["avg"].keys()
                avg_stats = {}
                for k in keys:
                    values = [float(g["avg"][k]) for g in gate_stats]
                    avg_stats[k] = sum(values) / len(values) if values else 0.0
            else:
                avg_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

            gate_summary = {"per_block": gate_stats, "avg": avg_stats}
            return x, final_states, all_states, patch_mask, captured, gate_summary
        else:
            return x, final_states, all_states, patch_mask, captured, {}


# -------------------------------------------------------------------
#  BiPMA - Bidirectional PMA wrapper
# -------------------------------------------------------------------
