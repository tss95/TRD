import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import logging
from typing import Optional, Tuple, Dict, Literal

# Import SwiGLU from stems.common
from trd.hosts.pmt.common import SwiGLU

logger = logging.getLogger(__name__)

_ATTENTION_BACKEND_OVERRIDE: Optional[str] = None
_FLASH_MASK_LEGACY_OVERRIDE: Optional[bool] = None
_FLASH_MASK_DEBUG_OVERRIDE: Optional[bool] = None
_FLASH_MASK_STATS_OVERRIDE: Optional[bool] = None
_FLASH_MASK_ASSERT_OVERRIDE: Optional[bool] = None
_DEBUG_NUMERICS_OVERRIDE: Optional[bool] = None
_FLASH_MASK_DEBUG_EMITTED = False
_FLASH_MASK_STATS: Dict[str, int] = {"calls": 0, "rows": 0, "blocked_rows": 0, "one_key_rows": 0}
_ROPE_INV_FREQ_CACHE: Dict[Tuple[str, int, int, float], torch.Tensor] = {}


def _env_flag(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def set_attention_backend(backend: Optional[str]) -> None:
    """Override attention backend selection (sdpa|math). Pass None to clear."""
    global _ATTENTION_BACKEND_OVERRIDE
    if backend is None:
        _ATTENTION_BACKEND_OVERRIDE = None
        return
    value = str(backend).strip().lower()
    if value in {"", "auto", "none", "default"}:
        _ATTENTION_BACKEND_OVERRIDE = None
        return
    _ATTENTION_BACKEND_OVERRIDE = value


def _update_flash_mask_stats(blocked: torch.Tensor, one_key: torch.Tensor, mask: torch.Tensor) -> None:
    global _FLASH_MASK_STATS
    rows = int(mask.numel() // max(mask.size(-1), 1))
    _FLASH_MASK_STATS["calls"] += 1
    _FLASH_MASK_STATS["rows"] += rows
    _FLASH_MASK_STATS["blocked_rows"] += int(blocked.sum().item())
    _FLASH_MASK_STATS["one_key_rows"] += int(one_key.sum().item())


def _flash_mask_legacy_enabled() -> bool:
    if _FLASH_MASK_LEGACY_OVERRIDE is not None:
        return _FLASH_MASK_LEGACY_OVERRIDE
    return bool(_env_flag("PMT_FLASH_MASK_LEGACY") or _env_flag("PMT_SANITISE_FLASH_MASK_LEGACY"))


def _flash_mask_debug_enabled() -> bool:
    if _FLASH_MASK_DEBUG_OVERRIDE is not None:
        return _FLASH_MASK_DEBUG_OVERRIDE
    return bool(_env_flag("PMT_FLASH_MASK_DEBUG"))


def _flash_mask_stats_enabled() -> bool:
    if _FLASH_MASK_STATS_OVERRIDE is not None:
        return _FLASH_MASK_STATS_OVERRIDE
    return bool(_env_flag("PMT_FLASH_MASK_STATS"))


def _flash_mask_assert_enabled() -> bool:
    if _FLASH_MASK_ASSERT_OVERRIDE is not None:
        return _FLASH_MASK_ASSERT_OVERRIDE
    env_flag = _env_flag("PMT_FLASH_MASK_ASSERT")
    if env_flag is not None:
        return bool(env_flag)
    return _flash_mask_debug_enabled()


def _debug_numerics_enabled() -> bool:
    if _DEBUG_NUMERICS_OVERRIDE is not None:
        return _DEBUG_NUMERICS_OVERRIDE
    return bool(_env_flag("PMT_DEBUG_NUMERICS"))


def _force_fp32_sdpa_enabled() -> bool:
    env_force = _env_flag("PMT_FORCE_FP32_SDPA")
    if env_force is not None:
        return bool(env_force)
    return False


def _neighborhood_allvalid_fastpath_enabled() -> bool:
    """Enable all-valid pad-mask fast path in neighborhood attention."""
    env_flag = _env_flag("PMT_NEIGHBORHOOD_ALLVALID_FASTPATH")
    if env_flag is not None:
        return bool(env_flag)
    return True


def _log_flash_mask_debug(
    *,
    mask: torch.Tensor,
    blocked_before: torch.Tensor,
    blocked_after: torch.Tensor,
    one_key: torch.Tensor,
    legacy: bool,
) -> None:
    global _FLASH_MASK_DEBUG_EMITTED
    if _FLASH_MASK_DEBUG_EMITTED:
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    _FLASH_MASK_DEBUG_EMITTED = True
    logger.info(
        "flash_mask_sanitiser legacy=%s shape=%s dtype=%s blocked_before=%d blocked_after=%d one_key=%d",
        legacy,
        tuple(mask.shape),
        mask.dtype,
        int(blocked_before.sum().item()),
        int(blocked_after.sum().item()),
        int(one_key.sum().item()),
    )


# Add RMSNorm availability check
try:
    from torch.nn import RMSNorm  # Available in PyTorch 2.0+

    RMSNORM_AVAILABLE = True
except ImportError:
    RMSNORM_AVAILABLE = False


# Add choose_norm utility function
def choose_norm(norm_type: str, d_model: int):
    """
    Returns either LayerNorm or RMSNorm (if available) based on norm_type="ln" or "rms".
    If "rms" is chosen but RMSNorm is unavailable, raises an error.
    """
    norm_type = norm_type.lower()
    if norm_type == "ln":
        return nn.LayerNorm(d_model, eps=1e-5)
    elif norm_type == "rms":
        if RMSNORM_AVAILABLE:
            return RMSNorm(d_model, eps=1e-5)
        else:
            raise ValueError("RMSNorm requested but not available in this PyTorch version.")
    else:
        raise ValueError(f"Unknown norm_type='{norm_type}'. Use 'ln' or 'rms'.")


################################################################################
# Utility function for Sanitising Masks for FlashAttention
################################################################################
def sanitise_flash_mask(
    mask: Optional[torch.Tensor], *, policy: Literal["self_attn", "causal", "prefix"] = "self_attn"
) -> Optional[torch.Tensor]:
    """
    Sanitise a boolean attention mask for Flash-2.

    policy:
      - "self_attn": standard self-attention; diagonal self-keys are legal.
      - "causal": square causal self-attention; do not open future keys.
      - "prefix": prefix/cross layouts; no diagonal assumptions or fallback openings.
    """
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean (True = block)")
    if mask.ndim < 2:
        raise ValueError(f"Mask must have at least 2 dims, got {mask.ndim}")

    mask = mask.clone()

    if policy == "prefix":
        fully_blocked = mask.all(dim=-1)
        if bool(fully_blocked.any()):
            raise RuntimeError(
                "sanitise_flash_mask(policy='prefix'): fully blocked rows present. "
                "Caller must construct a mask with at least one legal key per row for prefix/cross-attention layouts."
            )
        if _flash_mask_stats_enabled():
            one_key = (~mask).sum(dim=-1) == 1
            _update_flash_mask_stats(fully_blocked, one_key, mask)
        return mask

    if policy == "causal":
        q_len, k_len = mask.size(-2), mask.size(-1)
        if q_len != k_len:
            raise ValueError(
                "sanitise_flash_mask(policy='causal') supports only square causal self-attention (Q == K). "
                f"Got Q={q_len}, K={k_len}."
            )

        diag = torch.arange(q_len, device=mask.device)
        mask[..., diag, diag] = False
        fully_blocked = mask.all(dim=-1)
        if bool(fully_blocked.any()):
            raise RuntimeError(
                "sanitise_flash_mask(policy='causal'): fully blocked rows remain after opening the diagonal. "
                "Opening additional keys would violate causality. Fix the caller's mask/padding contract."
            )

        one_key = (~mask).sum(dim=-1) == 1
        mask[..., :, 0] = mask[..., :, 0] & (~one_key)
        if _flash_mask_stats_enabled():
            _update_flash_mask_stats(fully_blocked, one_key, mask)
        return mask

    if policy != "self_attn":
        raise ValueError(f"Unknown flash-mask sanitiser policy {policy!r}.")

    legacy = _flash_mask_legacy_enabled()
    debug = _flash_mask_debug_enabled()
    q_len, k_len = mask.size(-2), mask.size(-1)
    diag_len = min(q_len, k_len)
    diag = torch.arange(diag_len, device=mask.device)
    mask[..., diag, diag] = False

    blocked_before = mask.all(dim=-1)
    if legacy and mask.ndim == 2:
        mask[blocked_before, -1] = False
    else:
        mask[..., :, -1] = mask[..., :, -1] & (~blocked_before)

    one_key = (~mask).sum(dim=-1) == 1
    mask[..., :, 0] = mask[..., :, 0] & (~one_key)
    still_one_key = (~mask).sum(dim=-1) == 1
    if mask.size(-1) > 1:
        mask[..., :, 1] = mask[..., :, 1] & (~still_one_key)

    if _flash_mask_stats_enabled():
        _update_flash_mask_stats(blocked_before, one_key, mask)

    need_assert = _flash_mask_assert_enabled()
    blocked_after = mask.all(dim=-1) if (debug or need_assert) else None
    if debug and blocked_after is not None:
        _log_flash_mask_debug(
            mask=mask, blocked_before=blocked_before, blocked_after=blocked_after, one_key=one_key, legacy=legacy
        )

    if need_assert and blocked_after is not None and blocked_after.any():
        raise RuntimeError("sanitise_flash_mask: row still fully blocked after sanitisation.")
    return mask


################################################################################
# Utility function for Boolean to Additive Mask Conversion
################################################################################
def bool_to_additive(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Flash-2 bug-work-around: convert Bool[..., Q, K] (True = block) to
    additive bias of the same dtype as q/k/v, using appropriate large negative
    values based on dtype to avoid underflow issues.

    Uses:
    - -6e4 for float16/bfloat16 to avoid underflow
    - -1e9 for float32 for stronger masking
    """
    if mask.dtype != torch.bool:
        raise ValueError(f"Input mask must be boolean, got {mask.dtype}")

    # Choose appropriate negative value based on dtype
    neg_val = -6e4 if dtype in {torch.float16, torch.bfloat16} else -1e9

    # Use zeros_like for clearer type handling
    additive_mask = torch.where(
        mask, torch.full((), neg_val, dtype=dtype, device=mask.device), torch.zeros_like(mask, dtype=dtype)
    )
    return additive_mask


def _get_rope_inv_freq(half_dim: int, base: float, device: torch.device) -> torch.Tensor:
    """Return cached RoPE inverse frequencies on the target device."""
    key = (device.type, -1 if device.index is None else int(device.index), int(half_dim), float(base))
    inv_freq = _ROPE_INV_FREQ_CACHE.get(key)
    if inv_freq is None or inv_freq.device != device:
        freq_seq = torch.arange(half_dim, dtype=torch.float32, device=device)
        inv_freq = base ** (-freq_seq / half_dim)
        _ROPE_INV_FREQ_CACHE[key] = inv_freq
    return inv_freq


################################################################################
# Utility function for Rotary Position Embeddings (RoPE)
################################################################################
def apply_rope(q, k, positions, base=10000.0):
    """
    Apply Rotary Position Embeddings (RoPE) to query and key tensors.

    RoPE encodes relative positions by rotating vectors in the embedding space,
    which allows the model to generalize better to unseen sequence lengths.

    This implementation selectively applies RoPE based on position values:
    - For positions >= 0: Apply rotation based on the absolute position
    - For positions < 0: Skip rotation (useful for special tokens like state tokens)

    Args:
        q: Query tensor of shape [seq_len, batch_size, dim]
        k: Key tensor of shape [seq_len, batch_size, dim]
        positions: Position tensor of shape [seq_len] containing position indices
                  Use negative values to indicate tokens that should not receive RoPE
        base: Base for the frequency calculations (default: 10000.0)

    Returns:
        q_rotated: Rotated query tensor [seq_len, batch_size, dim]
        k_rotated: Rotated key tensor [seq_len, batch_size, dim]

    Notes:
        - The dimension (dim) must be even
        - Positions < 0 keep their original embeddings without rotation.
        - Based on the RoPE method from "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    debug_numerics = _debug_numerics_enabled()

    # Check inputs for NaN values
    # logger.info(f"apply_rope => q.shape={q.shape}, k.shape={k.shape}")
    if debug_numerics and (torch.isnan(q).any() or torch.isnan(k).any()):
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_distributed else 0
        if rank == 0:
            logger.warning("NaN detected in input to apply_rope. Attempting to fix.")
        q = torch.nan_to_num(q, nan=0.0, posinf=1.0, neginf=-1.0)
        k = torch.nan_to_num(k, nan=0.0, posinf=1.0, neginf=-1.0)

    seq_len, batch_size, dim = q.shape
    if dim % 2 != 0:
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_distributed else 0
        if rank == 0:
            logger.error(f"RoPE requires even dimensions, got dim={dim}")
        raise ValueError(f"RoPE requires even dimensions, got dim={dim}")

    try:
        # Keep RoPE numerically stable by computing frequencies in fp32, but
        # preserve the incoming activation dtype on the returned Q/K tensors.
        q_dtype = q.dtype
        k_dtype = k.dtype
        if positions.device != q.device:
            positions = positions.to(device=q.device)

        half_dim = dim // 2

        # Generate inverse frequency bands once per (device, dim, base).
        inv_freq = _get_rope_inv_freq(half_dim, base, q.device)  # [half_dim]

        # Handle negative positions (clamp to 0 for calculation, but will mask later)
        pos_clamped = torch.clamp(positions, min=0).to(dtype=torch.float32)  # [seq_len]

        # Calculate per-position frequency
        freqs = pos_clamped.unsqueeze(-1) * inv_freq.unsqueeze(0)  # [seq_len, half_dim]

        # Calculate sin and cos values, handle potential instability
        sin_vals = torch.sin(freqs)  # [seq_len, half_dim]
        cos_vals = torch.cos(freqs)  # [seq_len, half_dim]

        # Check for NaN in sin/cos (shouldn't happen but just in case)
        if debug_numerics and (torch.isnan(sin_vals).any() or torch.isnan(cos_vals).any()):
            is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
            rank = torch.distributed.get_rank() if is_distributed else 0
            if rank == 0:
                logger.warning("NaN detected in sin/cos values during RoPE. Fixing.")
            sin_vals = torch.nan_to_num(sin_vals, nan=0.0, posinf=0.0, neginf=0.0)
            cos_vals = torch.nan_to_num(cos_vals, nan=1.0, posinf=1.0, neginf=1.0)

        # Split q and k into first and second half along the last dimension
        q1, q2 = q[..., :half_dim], q[..., half_dim:]  # each [seq_len, batch_size, half_dim]
        k1, k2 = k[..., :half_dim], k[..., half_dim:]

        # Expand sin and cos for broadcasting and cast to match Q/K dtypes.
        sin_vals_q = sin_vals.to(dtype=q_dtype).unsqueeze(1)  # [seq_len, 1, half_dim]
        cos_vals_q = cos_vals.to(dtype=q_dtype).unsqueeze(1)  # [seq_len, 1, half_dim]
        if k_dtype == q_dtype:
            sin_vals_k, cos_vals_k = sin_vals_q, cos_vals_q
        else:
            sin_vals_k = sin_vals.to(dtype=k_dtype).unsqueeze(1)
            cos_vals_k = cos_vals.to(dtype=k_dtype).unsqueeze(1)

        # Use a bool mask for branchless select without extra arithmetic tensors.
        valid_mask = (positions >= 0).unsqueeze(-1).unsqueeze(-1)  # [L, 1, 1]

        # Apply rotation formula: [x_1, x_2] -> [x_1*cos - x_2*sin, x_1*sin + x_2*cos]
        q1_rotated = q1 * cos_vals_q - q2 * sin_vals_q
        q2_rotated = q1 * sin_vals_q + q2 * cos_vals_q

        k1_rotated = k1 * cos_vals_k - k2 * sin_vals_k
        k2_rotated = k1 * sin_vals_k + k2 * cos_vals_k

        # For positions < 0, keep original values.
        q1_final = torch.where(valid_mask, q1_rotated, q1)
        q2_final = torch.where(valid_mask, q2_rotated, q2)
        k1_final = torch.where(valid_mask, k1_rotated, k1)
        k2_final = torch.where(valid_mask, k2_rotated, k2)

        # Concatenate halves back together
        q_rotated = torch.cat([q1_final, q2_final], dim=-1)
        k_rotated = torch.cat([k1_final, k2_final], dim=-1)

        # Final NaN check
        if debug_numerics and (torch.isnan(q_rotated).any() or torch.isnan(k_rotated).any()):
            is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
            rank = torch.distributed.get_rank() if is_distributed else 0
            if rank == 0:
                logger.warning("NaN detected in output of apply_rope. Fixing.")
            q_rotated = torch.nan_to_num(q_rotated, nan=0.0, posinf=1.0, neginf=-1.0)
            k_rotated = torch.nan_to_num(k_rotated, nan=0.0, posinf=1.0, neginf=-1.0)

        if q_rotated.dtype != q_dtype:
            q_rotated = q_rotated.to(dtype=q_dtype)
        if k_rotated.dtype != k_dtype:
            k_rotated = k_rotated.to(dtype=k_dtype)
        return q_rotated, k_rotated

    except Exception as e:
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_distributed else 0
        if rank == 0:
            logger.error(f"Error in apply_rope: {str(e)}")
        raise RuntimeError(f"apply_rope failed: {e}") from e


def _apply_rope_to_projected_qk_per_head(
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_heads: int,
    d_head: int,
    rope_base: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE per-head to projected Q/K tensors.

    Args:
        q_proj: [B, L, D]
        k_proj: [B, L, D]
        positions: [L]
    Returns:
        Tuple of rotated (q_proj, k_proj), each [B, L, D].
    """
    if q_proj.shape != k_proj.shape:
        raise ValueError(f"Q/K projection shapes must match, got {tuple(q_proj.shape)} and {tuple(k_proj.shape)}")
    if q_proj.ndim != 3:
        raise ValueError(f"Q/K projections must be rank-3 [B,L,D], got rank {q_proj.ndim}")
    if positions.ndim != 1:
        raise ValueError(f"positions must be rank-1 [L], got shape {tuple(positions.shape)}")

    B, L, D = q_proj.shape
    if positions.numel() != L:
        raise ValueError(f"positions length ({positions.numel()}) must match sequence length ({L})")
    if D != num_heads * d_head:
        raise ValueError(
            f"Projected dim mismatch: D={D}, num_heads={num_heads}, d_head={d_head}, "
            f"num_heads*d_head={num_heads * d_head}"
        )

    # [B, L, D] -> [L, B*H, d_head]
    q_lbh = q_proj.reshape(B, L, num_heads, d_head).permute(1, 0, 2, 3).reshape(L, B * num_heads, d_head)
    k_lbh = k_proj.reshape(B, L, num_heads, d_head).permute(1, 0, 2, 3).reshape(L, B * num_heads, d_head)

    q_rope, k_rope = apply_rope(q_lbh, k_lbh, positions, base=rope_base)

    # [L, B*H, d_head] -> [B, L, D]
    q_out = q_rope.reshape(L, B, num_heads, d_head).permute(1, 0, 2, 3).reshape(B, L, D).contiguous()
    k_out = k_rope.reshape(L, B, num_heads, d_head).permute(1, 0, 2, 3).reshape(B, L, D).contiguous()
    return q_out, k_out


# -----------------------------------------------------------------------------
#  Multi-Head Attention with FlashAttention-2 via SDPA (PyTorch >= 2.1)
# -----------------------------------------------------------------------------
class MultiHeadAttentionFlash(nn.Module):
    """
    Drop-in replacement for MultiHeadAttention using SDPA (Flash-2).

    Parameters
    ----------
    dropout : float
       Nominal dropout probability (same field as before).
    dropout_mode : {"auto","legacy"}, default "auto"
       Retained for API compatibility; SDPA and math backends both apply
       attention dropout at the attention-probability site.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, dropout_mode: str = "auto"):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model, self.num_heads = d_model, num_heads
        self.d_head = d_model // num_heads
        # Use a single projection layer for Q, K, V for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)

        self.p_orig = float(dropout)
        self.mode = dropout_mode.lower()

        # only used in "legacy" mode
        self.drop_out = nn.Dropout(self.p_orig)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:  # Renamed argument for clarity
        # Input: [B, L, D] -> Output: [B, nH, L, d_head]
        B, L, _ = x.shape
        # Reshape and transpose for multi-head format expected by SDPA
        # Use reshape instead of view for safety with non-contiguous tensors
        return x.reshape(B, L, self.num_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, nH, L, d_head] -> Output: [B, L, D]
        B, nH, L, d = x.shape
        # Transpose back and reshape to original format
        # Add .contiguous() before reshape
        return x.transpose(1, 2).contiguous().reshape(B, L, nH * d)

    def project_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projects input x to Q, K, V tensors."""
        qkv = self.qkv_proj(x)  # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)  # Each [B, L, D]
        return q, k, v

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass using F.scaled_dot_product_attention.

        Args:
            q, k, v: Query, Key, Value tensors *after* projection, potentially after RoPE. Shape [B, L, D].
            attn_mask: Optional boolean attention mask [L, L] where **True indicates position is blocked/masked**.
            is_causal: If True, applies causal masking (takes precedence over attn_mask if provided).

        Returns:
            Output tensor of shape [B, L, D]
        """
        B, L, _ = q.shape

        # Split heads before passing to SDPA
        q = self._split_heads(q)  # [B, nH, L, d_head]
        k = self._split_heads(k)  # [B, nH, L, d_head]
        v = self._split_heads(v)  # [B, nH, L, d_head]

        mask_policy = None
        # If both is_causal and attn_mask are provided, merge in boolean space first.
        if is_causal and attn_mask is not None:
            L = q.size(-2)
            causal = torch.ones(L, L, dtype=torch.bool, device=q.device).triu(1)  # True = block
            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask | causal
                mask_policy = "causal"
            else:
                # Assume additive mask; merge by adding a causal additive mask.
                causal_add = bool_to_additive(causal, attn_mask.dtype)
                attn_mask = attn_mask + causal_add
            is_causal = False  # hand causal info via the merged mask

        # Convert boolean mask (True=block) to additive bias for SDPA.
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            if mask_policy is None:
                mask_policy = "causal" if is_causal else "self_attn"
            attn_mask = sanitise_flash_mask(attn_mask, policy=mask_policy)
            attn_mask = bool_to_additive(attn_mask, q.dtype)  # Use q's dtype

        # Make mask shape broadcastable to (B, nH, Q, K)
        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)  # [B,1,Q,K]

        # Precision-safety: run SDPA in fp32 when the model is bf16/fp16
        # (can be disabled for benchmarking via PMT_FORCE_FP32_SDPA=0)
        orig_dtype = q.dtype
        if _force_fp32_sdpa_enabled() and orig_dtype in (torch.float16, torch.bfloat16):
            q = q.float()
            k = k.float()
            v = v.float()
            if attn_mask is not None and torch.is_floating_point(attn_mask):
                attn_mask = attn_mask.float()

        attn_dropout_p = self.p_orig if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=attn_dropout_p, is_causal=is_causal
        )

        # Cast back to the original AMP dtype if we up-casted
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)

        out = self._merge_heads(out)  # [B, L, D]

        return self.o_proj(out)


# -----------------------------------------------------------------------------
#  Math (non-SDPA) Multi-Head Attention with the same API as Flash
# -----------------------------------------------------------------------------
class MultiHeadAttentionMath(nn.Module):
    """
    Drop-in replacement for MultiHeadAttentionFlash that computes attention
    using explicit matmul + softmax (no torch.scaled_dot_product_attention).

    Keeps the same projection helpers so encoder layers can remain unchanged.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, dropout_mode: str = "auto"):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model, self.num_heads = d_model, num_heads
        self.d_head = d_model // num_heads
        # Match Flash class projections for drop-in compatibility
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)
        self.p_orig = float(dropout)
        self.drop_out = nn.Dropout(self.p_orig)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.reshape(B, L, self.num_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, nH, L, d = x.shape
        return x.transpose(1, 2).contiguous().reshape(B, L, nH * d)

    def project_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.qkv_proj(x)  # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # Split heads
        q = self._split_heads(q)  # [B,H,Q,Dh]
        k = self._split_heads(k)  # [B,H,K,Dh]
        v = self._split_heads(v)  # [B,H,K,Dh]

        B, H, Q, Dh = q.shape
        K = k.size(-2)

        # Compute scaled scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # [B,H,Q,K]

        # Build/merge causal mask if requested
        if is_causal:
            causal = torch.ones(Q, K, dtype=torch.bool, device=scores.device).triu(1)
            scores = scores.masked_fill(causal, float('-inf'))

        # Apply attention mask if provided
        if attn_mask is not None:
            # Support both additive and boolean masks; broadcast to [B,1,Q,K]
            if attn_mask.dtype == torch.bool:
                m = attn_mask
                if m.dim() == 2:
                    m = m.unsqueeze(0).unsqueeze(0)  # [1,1,Q,K]
                elif m.dim() == 3:
                    m = m.unsqueeze(1)  # [B,1,Q,K]
                scores = scores.masked_fill(m, float('-inf'))
            else:
                # additive mask
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(1)
                scores = scores + attn_mask

        # Softmax + dropout
        attn = torch.softmax(scores, dim=-1)
        if self.training and self.p_orig > 0:
            attn = self.drop_out(attn)

        out = torch.matmul(attn, v)  # [B,H,Q,Dh]
        out = self._merge_heads(out)  # [B,Q,D]
        return self.o_proj(out)


def _choose_mha_impl(dropout_mode: str) -> nn.Module:
    """Select the attention backend via env PMT_ATTENTION_BACKEND={sdpa|math}."""
    backend = _ATTENTION_BACKEND_OVERRIDE or os.environ.get('PMT_ATTENTION_BACKEND', 'sdpa') or 'sdpa'
    backend = backend.strip().lower()
    if backend in ('math', 'vanilla', 'manual'):
        return MultiHeadAttentionMath
    return MultiHeadAttentionFlash


# -----------------------------------------------------------------------------
#  Original Multi-Head Attention (kept for reference/fallback if needed)
# -----------------------------------------------------------------------------


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, activation='relu', dropout=0.1):
        """
        Position-wise Feed-Forward Network.

        Args:
            d_model (int): Dimension of the model.
            d_ff (int): Dimension of the feed-forward network.
            activation (str): Activation function ('relu' or 'gelu').
            dropout (float): Dropout rate.
        """
        super(PositionWiseFeedForward, self).__init__()
        self.activation_str = activation.lower()
        if self.activation_str == "relu":
            # Use out-of-place ReLU to avoid autograd in-place modification issues
            self.activation = nn.ReLU()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
        elif self.activation_str == "gelu":
            self.activation = nn.GELU()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
        elif self.activation_str == "swiglu":
            # SwiGLU(d_model, d_ff) -> Linear(d_ff, d_model)
            self.swiglu_layer = SwiGLU(d_model, d_ff, init_xavier=True)
            self.linear_out = nn.Linear(d_ff, d_model)
        else:
            raise ValueError("Unsupported activation type. Choose 'relu', 'gelu', or 'swiglu'.")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for position-wise feed-forward network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.activation_str in ['relu', 'gelu']:
            # Apply the first linear layer ONLY for relu/gelu
            x = self.linear1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear2(x)
        elif self.activation_str == 'swiglu':
            # SwiGLU FFN: SwiGLU -> Linear
            # SwiGLU itself handles the d_model -> d_ff projection
            x = self.swiglu_layer(x)  # SwiGLU(d_model, d_ff)
            x = self.dropout(x)  # Dropout after SwiGLU
            x = self.linear_out(x)  # Linear(d_ff, d_model)
        else:
            # This case should not be reached due to init check
            raise RuntimeError(f"Invalid activation_str: {self.activation_str}")
        x = self.dropout(x)
        return x


class TransformerEncoderLayerNeighborhood(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        activation,
        dropout,
        neighborhood_size,
        norm_type="ln",
        norm_position="pre",
        name="TransformerEncoderLayerNeighborhood",
        dropout_mode="auto",
        has_cls_token=True,
    ):
        """
        Single Transformer Encoder Layer with Neighborhood Masking.
        We can switch between pre-norm and post-norm based on `norm_position`.

        Assumptions:
        - By default, a CLS token is at index 0 and can attend to all tokens.
        - With ``has_cls_token=False``, every position is an ordinary timestamp.
        - Ordinary tokens attend only to their causal neighborhood (t-n to t).

        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feed-forward network.
            activation (str): Activation function ('relu' or 'gelu').
            dropout (float): Dropout rate.
            neighborhood_size (int): Neighborhood size for masking.
            norm_type (str): Normalization type ('ln' for LayerNorm or 'rms' for RMSNorm).
            norm_position (str): Normalization position ('pre' or 'post').
            name (str): Name of the encoder layer.
            dropout_mode (str): Dropout mode ("auto" or "legacy") for attention.
            has_cls_token (bool): Whether sequence position zero is a global CLS token.
                Set false only for coordinate-preserving token streams that contain no CLS.
        """
        super(TransformerEncoderLayerNeighborhood, self).__init__()
        self.name = name
        MHA = _choose_mha_impl(dropout_mode)
        self.self_attn = MHA(d_model, num_heads, dropout, dropout_mode=dropout_mode)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, activation, dropout)

        # Use choose_norm for normalization layers
        self.norm1 = choose_norm(norm_type, d_model)
        self.norm2 = choose_norm(norm_type, d_model)

        self.dropout = nn.Dropout(dropout)
        self.neighborhood_size = neighborhood_size
        self.norm_position = norm_position.lower()  # "pre" or "post"
        self.has_cls_token = bool(has_cls_token)
        self._cached_mask_key = None
        self._cached_mask_raw = None
        self._cached_mask_sanitized = None

    def generate_unified_mask(self, seq_length, device, n_neighbors=None):
        """
        Generates a **boolean** local-causal neighborhood mask.
        - With a CLS token, index 0 can attend to all tokens.
        - Otherwise index 0 remains an ordinary causal timestamp.
        - Ordinary tokens attend to their neighborhood (t-n to t).

        Args:
            seq_length (int): Sequence length.
            device (torch.device): Device of the mask.
            n_neighbors (int, optional): Neighborhood size. Defaults to self.neighborhood_size.

        Returns:
            torch.Tensor: Boolean mask tensor of shape [1, 1, seq_length, seq_length].
                          **True indicates the position should be masked (blocked).**
        """
        if n_neighbors is None:
            n_neighbors = self.neighborhood_size

        q = torch.arange(seq_length, device=device).unsqueeze(1)  # rows (queries)
        k = torch.arange(seq_length, device=device).unsqueeze(0)  # cols (keys)

        local_causal = (k <= q) & (k >= (q - n_neighbors))
        base = ((q == 0) | local_causal) if self.has_cls_token else local_causal

        # Convert to required format: True = masked/blocked
        mask = ~base

        # Expand to [1, 1, L, L] for broadcasting in attention
        mask = mask.unsqueeze(0).unsqueeze(0)

        return mask

    def print_sample_mask(self, seq_length=6, n_neighbors=2):
        """
        Prints a sample mask for debugging purposes.

        Args:
            seq_length (int, optional): Sequence length for the sample mask. Defaults to 6.
            n_neighbors (int, optional): Neighborhood size for the sample mask. Defaults to 2.

        Note: Prints the ALLOWED (0) / BLOCKED (1) version for readability.
        """
        mask = self.generate_unified_mask(seq_length=seq_length, device='cpu', n_neighbors=n_neighbors)
        # Convert boolean mask (True=block) back to int (1=block) for printing
        mask_int = mask.int()
        print(f"\nSample Mask for {self.name} (CLS at start):")
        for row in mask_int.tolist():
            print(row)

    def forward(self, src, pad_mask=None, positions=None, use_rope=False, rope_base=10000.0):
        """
        Forward pass for the encoder layer with neighborhood masking, supporting both pre-norm and post-norm.

        Args:
            src (torch.Tensor): Input tensor of shape [batch_size, seq_length, d_model].
            pad_mask (torch.Tensor, optional): Padding mask of shape [batch_size, seq_length],
                                              where True indicates valid tokens and False indicates padding/NaN.
            positions (torch.Tensor, optional): RoPE positions [seq_length]. Ignored when use_rope=False.
            use_rope (bool): Whether to apply RoPE to Q/K projections in this layer.
            rope_base (float): RoPE frequency base.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_length, d_model].
        """
        batch_size, seq_length, d_model = src.size()
        device = src.device
        debug_numerics = _debug_numerics_enabled()

        # Generate and cache the geometry-only neighborhood mask.
        mask_key = (seq_length, str(device), self.neighborhood_size, self.has_cls_token)
        if self._cached_mask_key != mask_key:
            base_mask = self.generate_unified_mask(seq_length, device)
            self._cached_mask_key = mask_key
            self._cached_mask_raw = base_mask
            mask_policy = "self_attn" if self.has_cls_token else "causal"
            self._cached_mask_sanitized = sanitise_flash_mask(base_mask, policy=mask_policy)
        mask = self._cached_mask_raw

        use_cached_geometry_mask = pad_mask is None
        has_invalid_pad_tokens = False

        # Handle padding mask if provided.
        # Fast path: if every token is valid we can reuse the cached geometry mask and skip
        # constructing a dense [B,1,L,L] padding mask.
        if pad_mask is not None:
            pad_mask_bool = pad_mask.bool()
            if _neighborhood_allvalid_fastpath_enabled():
                all_valid = bool(torch.all(pad_mask_bool).item())
            else:
                all_valid = False

            if all_valid:
                use_cached_geometry_mask = True
            else:
                use_cached_geometry_mask = False
                has_invalid_pad_tokens = True

                # Convert pad_mask (True=valid) to boolean block mask (True=block)
                # Fix: Clone to ensure no shared storage with pad_mask
                pad_block = (~pad_mask_bool).clone()  # False=valid → True=block

                # Create row and column masks
                pad_row = pad_block.unsqueeze(2).clone()  # queries [B,L,1] - clone to allow in-place modification
                pad_col = pad_block.unsqueeze(1)  # keys [B,1,L]

                # Always allow the CLS row to attend when the sequence actually has one.
                if self.has_cls_token:
                    pad_row[:, 0, :] = False

                # Block if row OR col is invalid
                pad_mask_full = (pad_row | pad_col).clone()  # clone to allow in-place modification

                # Keep the CLS column open only in the historical CLS-bearing layout.
                if self.has_cls_token:
                    pad_mask_full[..., 0] = False

                # Combine with neighborhood mask (shape broadcast: [1,1,L,L] | [B,L,L])
                # First expand pad_mask_full to 4D
                pad_mask_full = pad_mask_full.unsqueeze(1)  # [B,1,L,L]
                mask = mask | pad_mask_full  # neighborhood OR padding

        if self.norm_position == "pre":
            # -------------------------
            #  Pre-Norm style
            # -------------------------
            # (1) Pre-norm + Self-Attention
            # Mask-consistent sanitization: zero-out invalid tokens before norm to avoid NaNs
            if has_invalid_pad_tokens:
                # pad_mask True=valid, False=invalid — clone to avoid shared storage/version bumps
                valid = pad_mask.bool().unsqueeze(-1).clone()  # [B,L,1]
                src = torch.where(valid, src, torch.zeros_like(src))
            x_norm = self.norm1(src)  # pre-LN

            # Project Q, K, V from normalized input
            q, k, v = self.self_attn.project_qkv(x_norm)
            if use_rope and positions is not None:
                q, k = _apply_rope_to_projected_qk_per_head(
                    q,
                    k,
                    positions,
                    num_heads=self.self_attn.num_heads,
                    d_head=self.self_attn.d_head,
                    rope_base=rope_base,
                )

            mask_policy = "self_attn" if self.has_cls_token else "causal"
            attn_mask = (
                self._cached_mask_sanitized
                if use_cached_geometry_mask
                else sanitise_flash_mask(mask, policy=mask_policy)
            )
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                attn_mask = bool_to_additive(attn_mask, q.dtype)
            attn_output = self.self_attn(q, k, v, attn_mask=attn_mask)

            # Only run expensive finite checks when numerics debugging is enabled.
            if debug_numerics and self.training:
                assert torch.isfinite(
                    attn_output
                ).all(), f"{self.name}: SDPA produced NaNs in pre-norm MHA - mask still bad?"

            src = src + self.dropout(attn_output)  # residual connection

            # (2) Pre-norm + Feed-Forward Network
            x_norm2 = self.norm2(src)  # pre-LN
            ff_output = self.feed_forward(x_norm2)  # [batch_size, seq_length, d_model]
            src = src + ff_output  # residual connection (no extra dropout - feed_forward already has dropout)
        else:
            # -------------------------
            #  Post-Norm style
            # -------------------------
            # (1) Self-Attention + Post-norm
            # Project Q, K, V from *original* input (apply same sanitization for safety)
            if has_invalid_pad_tokens:
                valid = pad_mask.bool().unsqueeze(-1).clone()
                src = torch.where(valid, src, torch.zeros_like(src))
            q, k, v = self.self_attn.project_qkv(src)
            if use_rope and positions is not None:
                q, k = _apply_rope_to_projected_qk_per_head(
                    q,
                    k,
                    positions,
                    num_heads=self.self_attn.num_heads,
                    d_head=self.self_attn.d_head,
                    rope_base=rope_base,
                )

            mask_policy = "self_attn" if self.has_cls_token else "causal"
            attn_mask = (
                self._cached_mask_sanitized
                if use_cached_geometry_mask
                else sanitise_flash_mask(mask, policy=mask_policy)
            )
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                attn_mask = bool_to_additive(attn_mask, q.dtype)
            attn_output = self.self_attn(q, k, v, attn_mask=attn_mask)

            # Only run expensive finite checks when numerics debugging is enabled.
            if debug_numerics and self.training:
                assert torch.isfinite(
                    attn_output
                ).all(), f"{self.name}: SDPA produced NaNs in post-norm MHA - mask still bad?"

            src = self.norm1(src + self.dropout(attn_output))  # residual connection + normalization

            # (2) Feed-Forward Network + Post-norm
            ff_output = self.feed_forward(src)  # [batch_size, seq_length, d_model]
            src = self.norm2(src + ff_output)  # residual connection + normalization (no extra dropout)

        # Debugging assertion - can be enabled for testing
        if __debug__ and False:  # Disabled by default, enable for debugging
            # Extract a simplified version for assertion
            sample_idx = 0 if mask.size(0) > 1 else 0  # First batch or only one
            mask_slice = mask[sample_idx] if mask.ndim > 3 else mask[0]
            fully_masked = mask_slice.all(dim=-1)
            assert not fully_masked.any(), "After combining masks, some rows are still fully masked - CLS not preserved"

        return src


class TransformerEncoderNeighborhood(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        activation='relu',
        dropout=0.1,
        neighborhood_size=5,
        norm_type="ln",
        norm_position="pre",
        debug=False,
        name="TransformerEncoderNeighborhood",
        dropout_mode="auto",
        has_cls_token=True,
        apply_terminal_norm: bool = True,
    ):
        """
        Transformer Encoder with Neighborhood Masking.

        Assumptions:
        - A CLS token is present at index 0 by default.
        - Set ``has_cls_token=False`` for coordinate-only streams with no CLS.

        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of encoder layers.
            d_ff (int): Dimension of the feed-forward network.
            activation (str): Activation function ('relu' or 'gelu').
            dropout (float): Dropout rate.
            neighborhood_size (int): Neighborhood size for masking.
            norm_type (str): Normalization type ('ln' for LayerNorm or 'rms' for RMSNorm).
            norm_position (str): Normalization position ('pre' or 'post').
            debug (bool): Debug flag to print sample masks.
            name (str): Name of the encoder.
            dropout_mode (str): Dropout mode ("auto" or "legacy") for attention.
            has_cls_token (bool): Whether sequence position zero is a global CLS token.
            apply_terminal_norm (bool): Apply a stack-terminal norm after all encoder layers.
                Disabling this leaves every layer-internal norm unchanged.
        """
        super(TransformerEncoderNeighborhood, self).__init__()
        self.name = name
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerNeighborhood(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    activation=activation,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    norm_type=norm_type,
                    norm_position=norm_position,
                    name=f"{name}_Layer{i+1}",
                    dropout_mode=dropout_mode,
                    has_cls_token=has_cls_token,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = choose_norm(norm_type, d_model) if apply_terminal_norm else nn.Identity()
        self.debug = debug

        # If debug is True, print a sample mask
        if self.debug:
            # Print for n_neighbors=3 and seq_length=5
            self.layers[0].print_sample_mask(seq_length=5, n_neighbors=3)

    def forward(self, src, pad_mask=None, positions=None, use_rope=False, rope_base=10000.0):
        """
        Forward pass for the transformer encoder.

        Args:
            src (torch.Tensor): Input tensor of shape [batch_size, seq_length, d_model].
                                Position zero is CLS only when ``has_cls_token=True``.
            pad_mask (torch.Tensor, optional): Padding mask of shape [batch_size, seq_length],
                                              where True indicates valid tokens and False indicates padding/NaN.
            positions (torch.Tensor, optional): RoPE positions [seq_length]. Ignored when use_rope=False.
            use_rope (bool): Whether to apply RoPE in neighborhood self-attention.
            rope_base (float): RoPE frequency base.

        Returns:
            torch.Tensor: Encoded tensor [batch_size, seq_length, d_model].
        """
        for layer in self.layers:
            src = layer(src, pad_mask=pad_mask, positions=positions, use_rope=use_rope, rope_base=rope_base)
        src = self.norm(src)
        return src


class TransformerEncoderLayerWithMask(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        activation='relu',
        dropout=0.1,
        norm_type="ln",
        norm_position="pre",
        name="TransformerEncoderLayerWithMask",
        dropout_mode="auto",
    ):
        """
        Transformer Encoder Layer that accepts an external attention mask.
        Supports both pre-norm and post-norm architectures based on norm_position.

        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the feed-forward network.
            activation (str): Activation function ('relu' or 'gelu').
            dropout (float): Dropout rate.
            norm_type (str): Normalization type ('ln' for LayerNorm or 'rms' for RMSNorm).
            norm_position (str): Normalization position ('pre' or 'post').
            name (str): Layer name for debugging/logging.
            dropout_mode (str): Dropout mode ("auto" or "legacy") for attention.

        Expected Input/Output shape: [L, B, D]
          - L: sequence length
          - B: batch size
          - D: feature dimension (d_model)
        """
        super(TransformerEncoderLayerWithMask, self).__init__()
        self.name = name
        self.norm_position = norm_position.lower()  # "pre" or "post"
        # Limit debug spam: only emit RoPE debug for first two forwards
        self._debug_forward_count = 0

        # Choose attention backend (sdpa|math) via env PMT_ATTENTION_BACKEND
        MHA = _choose_mha_impl(dropout_mode)
        self.self_attn = MHA(d_model, num_heads, dropout, dropout_mode=dropout_mode)

        # Position-wise feed-forward network
        activation_l = activation.lower()
        if activation_l == 'relu':
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            # Prefer non-inplace activations for safer autograd
            self.activation_fn = nn.ReLU()
            self.ffn_type = 'relu_gelu'
        elif activation_l == 'gelu':
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.activation_fn = nn.GELU()
            self.ffn_type = 'relu_gelu'
        elif activation_l == 'swiglu':
            # For SwiGLU, we need two linear layers for the gate and value
            # followed by a final linear layer. Replace linear1/act/linear2.
            # Use SwiGLU(d_model, d_ff) -> Linear(d_ff, d_model)
            self.swiglu_layer = SwiGLU(d_model, d_ff, init_xavier=True)
            self.linear_out = nn.Linear(d_ff, d_model)  # Final projection
            self.ffn_type = 'swiglu'
        else:
            raise ValueError("Unsupported activation. Choose 'relu', 'gelu', or 'swiglu'.")

        # Layer norm and dropout (using the utility function)
        self.norm1 = choose_norm(norm_type, d_model)
        self.norm2 = choose_norm(norm_type, d_model)
        self.dropout = nn.Dropout(dropout)
        self.supports_attn_mask_sanitized = True

    def feed_forward(self, x):
        """
        Forward pass of the position-wise feed-forward.
        x: [L, B, D]
        """
        if self.ffn_type == 'relu_gelu':
            # Standard FFN: Linear -> Activation -> Linear
            x = self.linear1(x)
            x = self.activation_fn(x)
            x = self.dropout(x)
            x = self.linear2(x)
        elif self.ffn_type == 'swiglu':
            # SwiGLU FFN: SwiGLU -> Linear
            # SwiGLU itself handles the d_model -> d_ff projection
            x = self.swiglu_layer(x)  # SwiGLU(d_model, d_ff)
            x = self.dropout(x)  # Dropout after SwiGLU
            x = self.linear_out(x)  # Linear(d_ff, d_model)
        else:
            # This case should not be reached due to init check
            raise RuntimeError(f"Invalid ffn_type: {self.ffn_type}")
        x = self.dropout(x)
        return x

    def forward(
        self, src, attn_mask=None, positions=None, use_rope=False, rope_base=10000.0, attn_mask_sanitized: bool = False
    ):
        """
        Forward pass through the transformer encoder layer with optional RoPE,
        supporting both pre-norm and post-norm styles.

        Args:
            src (torch.Tensor): Input tensor of shape [seq_len, batch_size, dim]
            attn_mask (torch.Tensor, optional): Boolean attention mask of shape [seq_len, seq_len]
                                               where True values are masked (blocked) positions
            positions (torch.Tensor, optional): Position indices for RoPE of shape [seq_len]
                                              Values >= 0 get rotational encoding, < 0 are skipped
            use_rope (bool): Whether to apply Rotary Position Embeddings (RoPE)
            rope_base (float): Base value for RoPE frequency calculations

        Returns:
            torch.Tensor: Transformed output of shape [seq_len, batch_size, dim]
        """
        # Get distributed rank for logging
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_distributed else 0

        # Debug logging for RoPE usage
        if use_rope and positions is not None and rank == 0 and self._debug_forward_count < 2:
            logger.debug(f"{self.name}: Using RoPE with {len(positions)} positions")
            self._debug_forward_count += 1

        debug_numerics = _debug_numerics_enabled()

        # Check for NaN in input
        if debug_numerics and torch.isnan(src).any():
            if rank == 0:
                logger.warning(f"{self.name}: NaN detected in input tensor")
                # Instead of crashing, we'll try to recover
                src = torch.nan_to_num(src, nan=0.0, posinf=1.0, neginf=-1.0)

        assert_finite(src, f"{self.name} - input 'src'")  # Check input

        src_transposed = src.transpose(0, 1)

        if self.norm_position == "pre":
            # --- Pre-Norm ---
            x_norm = self.norm1(src)  # 1. Norm
            q_proj, k_proj, v_proj = self.self_attn.project_qkv(
                x_norm.transpose(0, 1)
            )  # 2. Project QKV (from normalized)
            assert_finite(q_proj, f"{self.name} - q_proj (pre-RoPE)")  # Check projections
            assert_finite(k_proj, f"{self.name} - k_proj (pre-RoPE)")
            assert_finite(v_proj, f"{self.name} - v_proj")
            # 3. Apply RoPE (if needed)
            if use_rope and positions is not None:
                q_proj, k_proj = _apply_rope_to_projected_qk_per_head(
                    q_proj,
                    k_proj,
                    positions,
                    num_heads=self.self_attn.num_heads,
                    d_head=self.self_attn.d_head,
                    rope_base=rope_base,
                )
                assert_finite(q_proj, f"{self.name} - q_proj (post-RoPE)")  # Check after RoPE
                assert_finite(k_proj, f"{self.name} - k_proj (post-RoPE)")
            # 4. Attention
            if attn_mask is not None and not attn_mask_sanitized:
                attn_mask = sanitise_flash_mask(attn_mask, policy="self_attn")
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                attn_mask = bool_to_additive(attn_mask, q_proj.dtype)
            attn_output = self.self_attn(q_proj, k_proj, v_proj, attn_mask=attn_mask)
            assert_finite(attn_output, f"{self.name} - after attention")  # Check after attention
            if debug_numerics:
                assert torch.isfinite(
                    attn_output
                ).all(), f"{self.name}: SDPA produced NaNs in pre-norm layer - mask still bad?"
            # 5. Residual 1 (Attention)
            src = src + self.dropout(attn_output.transpose(0, 1))
            assert_finite(src, f"{self.name} - after residual 1")
            # 6. Norm 2 + 7. Feed Forward
            src_norm2 = self.norm2(src)
            assert_finite(src_norm2, f"{self.name} - after norm2")
            ff_output = self.feed_forward(src_norm2)
            assert_finite(ff_output, f"{self.name} - after feed_forward")
            # 8. Residual 2 (FFN)
            output = src + ff_output  # No extra dropout - feed_forward already has dropout
            assert_finite(output, f"{self.name} - final output (pre-norm)")
        else:  # Post-Norm
            # --- Post-Norm ---
            q_proj, k_proj, v_proj = self.self_attn.project_qkv(src_transposed)  # 1. Project QKV (from original)
            assert_finite(q_proj, f"{self.name} - q_proj (pre-RoPE)")  # Check projections
            assert_finite(k_proj, f"{self.name} - k_proj (pre-RoPE)")
            assert_finite(v_proj, f"{self.name} - v_proj")
            # 2. Apply RoPE (if needed)
            if use_rope and positions is not None:
                q_proj, k_proj = _apply_rope_to_projected_qk_per_head(
                    q_proj,
                    k_proj,
                    positions,
                    num_heads=self.self_attn.num_heads,
                    d_head=self.self_attn.d_head,
                    rope_base=rope_base,
                )
                assert_finite(q_proj, f"{self.name} - q_proj (post-RoPE)")  # Check after RoPE
                assert_finite(k_proj, f"{self.name} - k_proj (post-RoPE)")
            # 3. Attention
            if attn_mask is not None and not attn_mask_sanitized:
                attn_mask = sanitise_flash_mask(attn_mask, policy="self_attn")
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                attn_mask = bool_to_additive(attn_mask, q_proj.dtype)
            attn_output = self.self_attn(q_proj, k_proj, v_proj, attn_mask=attn_mask)
            assert_finite(attn_output, f"{self.name} - after attention")  # Check after attention
            if debug_numerics:
                assert torch.isfinite(
                    attn_output
                ).all(), f"{self.name}: SDPA produced NaNs in post-norm layer - mask still bad?"
            # 4. Residual 1 + Norm 1
            src = self.norm1(src + self.dropout(attn_output.transpose(0, 1)))
            assert_finite(src, f"{self.name} - after residual 1 + norm1")
            # 5. Feed Forward
            ff_output = self.feed_forward(src)
            assert_finite(ff_output, f"{self.name} - after feed_forward")
            # 6. Residual 2 + Norm 2
            output = self.norm2(src + ff_output)  # No extra dropout - feed_forward already has dropout
            assert_finite(output, f"{self.name} - final output (post-norm)")

        return output


class TransformerDecoderLayerCausal(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        activation='relu',
        dropout=0.1,
        name="TransformerDecoderLayerCausal",
        dropout_mode="auto",
    ):
        """
        Single Transformer Decoder Layer with causal self-attention only.

        This layer:
        - Performs causal self-attention on the input sequence.
        - Applies a position-wise feed-forward network.

        Args:
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
            d_ff (int): Feed-forward dimension.
            activation (str): 'relu' or 'gelu'.
            dropout (float): Dropout rate.
            name (str): Layer name.
            dropout_mode (str): Dropout mode ("auto" or "legacy") for attention.
        """
        super(TransformerDecoderLayerCausal, self).__init__()
        self.name = name

        # Choose attention backend (sdpa|math)
        MHA = _choose_mha_impl(dropout_mode)
        self.self_attn = MHA(d_model, num_heads, dropout, dropout_mode=dropout_mode)

        # Feed-forward network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, activation, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, pad_mask=None):
        """
        Forward pass with optional padding mask.

        Args:
            tgt (torch.Tensor): Input tensor of shape [batch_size, seq_length, d_model].
            pad_mask (torch.Tensor, optional): Padding mask of shape [batch_size, seq_length],
                                               where True indicates valid tokens and False indicates padding/NaN.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_length, d_model].
        """
        batch_size, seq_length, _ = tgt.size()

        # Get query, key, and value projections
        q, k, v = self.self_attn.project_qkv(tgt)

        # Build effective attention mask
        attn_mask = None

        # Handle padding mask if provided
        if pad_mask is not None:
            # Convert pad_mask (True=valid) to boolean block mask (True=block)
            block = ~pad_mask  # [B,L] True = block
            # Create row and column masks
            row = block.unsqueeze(2)  # [B,L,1]
            col = block.unsqueeze(1)  # [B,1,L]
            # Block if either query or key is padding
            attn_mask = row | col  # [B,L,L] bool, True = block
            # Keep the very last key open to placate Flash-2
            attn_mask[..., -1] = False

        # Build one mask that contains BOTH padding and causal constraints
        if attn_mask is None:  # only causal
            attn_out = self.self_attn(q, k, v, attn_mask=None, is_causal=True)
        else:  # padding + causal
            L = attn_mask.size(-1)
            causal = torch.ones(L, L, dtype=torch.bool, device=attn_mask.device).triu(1)  # True=block
            attn_mask = attn_mask | causal  # combine
            attn_mask = sanitise_flash_mask(attn_mask, policy="causal")
            attn_mask = bool_to_additive(attn_mask, q.dtype)
            attn_out = self.self_attn(q, k, v, attn_mask=attn_mask, is_causal=False)  # <-- ONLY ONE ARG

        # Add & norm (pre-norm)
        tgt = self.norm1(tgt + self.dropout(attn_out))

        # Feed-forward network
        ff_out = self.feed_forward(tgt)

        # Add & norm (pre-norm)
        tgt = self.norm2(tgt + self.dropout(ff_out))

        return tgt


class TransformerDecoderCausal(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        activation='relu',
        dropout=0.1,
        name="TransformerDecoderCausal",
        dropout_mode="auto",
    ):
        """
        Causal Transformer Decoder stack.

        Args:
            d_model (int): Model dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of decoder layers.
            d_ff (int): Feed-forward dimension.
            activation (str): Activation ('relu' or 'gelu').
            dropout (float): Dropout rate.
            name (str): Name of the decoder.
            dropout_mode (str): Dropout mode ("auto" or "legacy") for attention.
        """
        super(TransformerDecoderCausal, self).__init__()
        self.name = name
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayerCausal(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    activation=activation,
                    dropout=dropout,
                    name=f"{name}_Layer{i+1}",
                    dropout_mode=dropout_mode,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, pad_mask=None):
        """
        Forward pass through the decoder with optional padding mask.

        Args:
            tgt (torch.Tensor): Input tensor of shape [batch_size, seq_length, d_model].
            pad_mask (torch.Tensor, optional): Padding mask of shape [batch_size, seq_length],
                                               where True indicates valid tokens and False indicates padding/NaN.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_length, d_model].
        """
        for layer in self.layers:
            tgt = layer(tgt, pad_mask=pad_mask)

        return self.norm(tgt)


# Helper function for NaN/inf checking
def assert_finite(x, tag):
    if not _debug_numerics_enabled():
        return
    if not torch.isfinite(x).all():
        bad_count = (~torch.isfinite(x)).sum()
        total_count = x.numel()
        bad_percent = (bad_count / total_count) * 100
        # Get range ignoring NaNs for better info
        finite_vals = x[torch.isfinite(x)]
        min_val = finite_vals.min() if finite_vals.numel() > 0 else 'N/A'
        max_val = finite_vals.max() if finite_vals.numel() > 0 else 'N/A'
        raise RuntimeError(
            f"{tag}: {bad_count}/{total_count} ({bad_percent:.4f}%) non-finite values. Min: {min_val}, Max: {max_val}"
        )
