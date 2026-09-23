import logging
import os
from functools import lru_cache

import torch
import torch.distributed as dist

logger = logging.getLogger('PMT')

LOGIT_CLAMP = 100.0
GATHER_MAX_ELEMENTS = int(1e8)
PMT_CHECK_NUMERICS_ENV = "PMT_CHECK_NUMERICS"
GLOBAL_LOGIT_NUMERICS_CHECK = os.getenv(PMT_CHECK_NUMERICS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def set_logit_clamp(value: float) -> None:
    global LOGIT_CLAMP
    LOGIT_CLAMP = value


def safe_logits(
    mat: torch.Tensor, τ: torch.Tensor | float, debug: bool = False, check_numerics: bool | None = None
) -> torch.Tensor:
    """
    Safely compute logits in fp32 with clamping to prevent overflow.

    Args:
        mat: Matrix multiplication result before temperature scaling
        τ: Temperature to scale logits by (float or 0-D tensor)
        debug: If True, log min/max values
        check_numerics: If True, validate temperature positivity and finite outputs.
            This performs scalar reads (`.item()`/`bool(...)`) and can add sync overhead on CUDA.
            Keep this disabled on hot paths. If None, follows `debug` or
            the process-level env toggle `PMT_CHECK_NUMERICS`.

    Returns:
        Clamped logits in fp32

    Raises:
        RuntimeError: If logits contain NaN/Inf values or temperature ≤ 0
    """
    # Keep this out-of-place. If `mat` is already fp32, in-place ops here would mutate caller-owned tensors.
    logits = mat.float() if mat.dtype != torch.float32 else mat

    # Replace in-place operations with out-of-place ones
    logits = logits / τ  # was logits.div_(τ)
    logits = logits.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)

    if check_numerics is None:
        check_numerics = bool(debug) or bool(GLOBAL_LOGIT_NUMERICS_CHECK)

    if check_numerics:
        # Temperature must stay strictly positive.
        if isinstance(τ, torch.Tensor):
            if not torch.isfinite(τ).all().item():
                raise ValueError("temperature must be finite")
            if not (τ > 0).all().item():
                raise ValueError("temperature must be > 0")
        else:
            if τ <= 0:
                raise ValueError(f"temperature must be > 0, got {τ}")

        if not torch.isfinite(logits).all().item():
            raise RuntimeError("logits contain NaN/Inf")

    if debug and mat.dtype == torch.float16 and logits.numel() > 0:  # skip fp32→fp32 promotion in eval
        logger.debug(f"safe_logits: min={logits.min().item():.3f}, max={logits.max().item():.3f}")

    return logits


def unfold_windows(x: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    """
    Canonical windows: (B, Nw, window_size, D).
    Works on every PyTorch version from 1.13 to 2.3.
    """
    win = x.unfold(1, window_size, stride)
    return win.permute(0, 1, 3, 2).contiguous()


def assert_unfold_ok(x, win_size, stride):
    """
    Verify that unfolding produces the expected layout.

    Args:
        x: Input tensor of shape (B, L, D)
        win_size: Window size
        stride: Stride between windows
    """
    out = unfold_windows(x, win_size, stride)
    assert out.shape[-2] == win_size and out.shape[-1] == x.shape[-1], f"Unexpected unfold layout: got {out.shape}"


def _distributed_all_gather(local_tensor: torch.Tensor):
    """
    All-gather local_tensor from all ranks => one giant global tensor using a memory-efficient approach.
    Handles non-distributed case gracefully.

    Args:
        local_tensor (torch.Tensor): Tensor of shape (B_local, D) to gather from the current rank.

    Returns:
        Tuple[torch.Tensor, int, int]:
            - global_tensor: Concatenated tensor of shape (B_global, D).
            - local_offset: The starting row index for this rank's data in global_tensor.
            - global_size: The total number of rows (B_global).
    """
    device = local_tensor.device  # Define device here at the top

    # Check if distributed environment is available and initialized
    if not (dist.is_available() and dist.is_initialized()):
        B = local_tensor.size(0)
        return local_tensor, 0, B

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Ensure local_tensor is contiguous before flattening
    local_tensor = local_tensor.contiguous()
    local_flat = local_tensor.flatten()  # 1-D
    local_sz = torch.tensor([local_flat.numel()], device=device, dtype=torch.long)

    # Get original dimension for proper reshaping later
    original_dim = local_tensor.size(1) if local_tensor.dim() > 1 else 1

    # Gather and validate embedding dimensions across ranks
    if local_tensor.dim() > 1:
        dim_tensor = torch.tensor([original_dim], device=device, dtype=torch.long)
        dim_list = [torch.zeros_like(dim_tensor) for _ in range(world_size)]
        dist.all_gather(dim_list, dim_tensor)
        dims = [d.item() for d in dim_list]

        # Check for inconsistent dimensions
        if len(set(dims)) > 1:
            dims_str = ", ".join([f"rank {i}: {d}" for i, d in enumerate(dims)])
            raise ValueError(f"Inconsistent embedding dimensions across ranks: {dims_str}")

    size_list = [torch.zeros_like(local_sz) for _ in range(world_size)]
    dist.all_gather(size_list, local_sz)  # Gather 1-D sizes (elements)
    sizes = [int(s.item()) for s in size_list]  # List of element counts per rank

    if any(s < 0 for s in sizes):
        raise RuntimeError(
            f"_distributed_all_gather received negative sizes {sizes} for tensor shape {tuple(local_tensor.shape)}"
        )
    total_elements = int(sum(sizes))
    if total_elements <= 0:
        empty_shape = (0, original_dim) if original_dim > 1 else (0,)
        empty = torch.empty(empty_shape, dtype=local_tensor.dtype, device=device)
        return empty, 0, 0
    max_sz = int(max(sizes)) if sizes else 0
    if max_sz > GATHER_MAX_ELEMENTS or total_elements > GATHER_MAX_ELEMENTS:
        safe_rows_per_rank = 0
        if original_dim > 0 and world_size > 0:
            safe_rows_per_rank = int(GATHER_MAX_ELEMENTS // max(original_dim * world_size, 1))
        raise RuntimeError(
            f"_distributed_all_gather sizes unrealistically large "
            f"(max_size={max_sz}, total={total_elements}, sizes={sizes}, tensor_shape={tuple(local_tensor.shape)}, "
            f"guard_max_elements={GATHER_MAX_ELEMENTS}, world_size={world_size}, safe_rows_per_rank~={safe_rows_per_rank})"
        )
    if original_dim > 1 and any((s % original_dim) != 0 for s in sizes):
        raise RuntimeError(
            f"_distributed_all_gather got non-divisible sizes for 2D tensor "
            f"(dim={original_dim}, sizes={sizes}, tensor_shape={tuple(local_tensor.shape)})"
        )

    # Pad to max_sz so all_gather works
    if local_flat.numel() < max_sz:
        pad = torch.empty(max_sz - local_flat.numel(), dtype=local_flat.dtype, device=device)
        local_flat = torch.cat([local_flat, pad], 0)
        # No need for .contiguous() here, cat preserves contiguity

    gather_list = [torch.empty(max_sz, dtype=local_flat.dtype, device=device) for _ in range(world_size)]
    dist.all_gather(gather_list, local_flat)

    # Truncate & reshape each chunk
    chunks = []
    for i in range(world_size):
        if sizes[i] > 0:
            # Reshape back to (-1, D) or (-1,) for 1D tensors
            if original_dim > 1:
                reshaped_chunk = gather_list[i][: sizes[i]].view(-1, original_dim)
            else:
                reshaped_chunk = gather_list[i][: sizes[i]]  # Keep as 1D
            chunks.append(reshaped_chunk)

    if chunks:
        global_tensor = torch.cat(chunks, 0)
        # Cat preserves contiguity, no need for extra .contiguous()
    else:
        # Determine correct empty shape based on original dimensions
        empty_shape = (0, original_dim) if original_dim > 1 else (0,)
        global_tensor = torch.empty(empty_shape, dtype=local_tensor.dtype, device=device)

    # Calculate the offset based on the number of *elements* before this rank
    element_offset = sum(sizes[:rank])
    # Convert element offset to row offset (only if D > 1)
    local_offset = element_offset // original_dim if original_dim > 1 else element_offset

    # Return the total number of *rows* gathered (or elements if 1D)
    global_size = total_elements // original_dim if original_dim > 1 else total_elements

    return global_tensor, local_offset, global_size


def estimate_hgcl_patch_all_gather_budget(
    *,
    per_device_batch_size: int,
    token_seq_len: int,
    local_window_size: int,
    window_stride: int,
    proj_output_dim: int,
    world_size: int,
    max_total_elements: int = GATHER_MAX_ELEMENTS,
) -> dict[str, int]:
    """
    Estimate HGCL patch all-gather footprint from resolved training geometry.

    This mirrors the tensor built in HGCL before distributed gather:
      z2_patch_flat shape ~= (B_local * num_windows * local_window_size, proj_output_dim)

    Returns integer diagnostics suitable for logging and fail-fast checks.
    """
    per_device_batch_size = int(per_device_batch_size)
    token_seq_len = int(token_seq_len)
    local_window_size = int(local_window_size)
    window_stride = int(window_stride)
    proj_output_dim = int(proj_output_dim)
    world_size = max(int(world_size), 1)
    max_total_elements = int(max_total_elements)

    if per_device_batch_size < 0:
        raise ValueError(f"per_device_batch_size must be >= 0, got {per_device_batch_size}")
    if token_seq_len < 0:
        raise ValueError(f"token_seq_len must be >= 0, got {token_seq_len}")
    if local_window_size <= 0:
        raise ValueError(f"local_window_size must be > 0, got {local_window_size}")
    if window_stride <= 0:
        raise ValueError(f"window_stride must be > 0, got {window_stride}")
    if proj_output_dim <= 0:
        raise ValueError(f"proj_output_dim must be > 0, got {proj_output_dim}")
    if max_total_elements <= 0:
        raise ValueError(f"max_total_elements must be > 0, got {max_total_elements}")

    if token_seq_len < local_window_size:
        num_windows = 0
    else:
        num_windows = (token_seq_len - local_window_size) // window_stride + 1

    rows_per_rank = per_device_batch_size * num_windows * local_window_size
    elements_per_rank = rows_per_rank * proj_output_dim
    total_elements = elements_per_rank * world_size

    per_batch_denominator = num_windows * local_window_size * proj_output_dim
    if per_batch_denominator > 0:
        safe_per_device_batch_by_rank = max_total_elements // per_batch_denominator
        safe_per_device_batch_by_total = max_total_elements // (world_size * per_batch_denominator)
    else:
        # If there are no windows, gather tensor would be empty regardless of batch size.
        safe_per_device_batch_by_rank = 0
        safe_per_device_batch_by_total = 0

    safe_per_device_batch = min(safe_per_device_batch_by_rank, safe_per_device_batch_by_total)
    safe_effective_global_batch = safe_per_device_batch * world_size

    return {
        "num_windows": int(num_windows),
        "rows_per_rank": int(rows_per_rank),
        "elements_per_rank": int(elements_per_rank),
        "total_elements": int(total_elements),
        "safe_per_device_batch_by_rank": int(safe_per_device_batch_by_rank),
        "safe_per_device_batch_by_total": int(safe_per_device_batch_by_total),
        "safe_per_device_batch": int(safe_per_device_batch),
        "safe_effective_global_batch": int(safe_effective_global_batch),
    }


def all_gather_tensor(x: torch.Tensor) -> tuple:
    """
    Gather `x` (N_local, D) from all ranks into one contiguous tensor.
    Works on both NCCL (NVidia + ROCm) and GLOO backends.

    Args:
        x: Tensor to be gathered across ranks

    Returns:
        Tuple of (global_tensor, local_offset, global_rows)
    """
    return _distributed_all_gather(x)


# Make both names available for backward compatibility, while keeping one implementation.
distributed_all_gather = _distributed_all_gather


@lru_cache(maxsize=1)
def get_default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def __getattr__(name: str):
    if name == "device":
        return get_default_device()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
