"""Shared components for Progressive Memory Attention modules.

This module contains reusable components that are duplicated across different
PMA versions (v0, v2, v3, v4, v5) and sequential_transformer.

Components:
- GateModule: Learnable gate for mixing recurrent state tensors
- LocalOffsetEmbedding: Position embeddings for tokens within a window
- ChunkOffsetEmbedding: Global position embeddings for chunk indices
- SkipGate: Learnable gate for residual connections
- OverlapAggregator: Multi-head attention aggregator for overlapping windows
"""

import logging
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Callable, Dict


def push_gate_stats(*args, **kwargs):
    """Gate telemetry is omitted from the paper training runner."""
    return None


logger = logging.getLogger('PMT')


class ZeroInitLinear(nn.Linear):
    """Linear layer with deterministic zero-weight initialization.

    This avoids RNG consumption during module construction and initializes
    tokenwise gates to a controlled logit via bias.
    """

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, bias_init: float = 0.0) -> None:
        self._bias_init = float(bias_init)
        super().__init__(in_features, out_features, bias=bias)

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, self._bias_init)


def _gate_stats_dict(gate: torch.Tensor) -> Dict[str, float]:
    gate_f = gate.detach().float()
    return {
        "mean": float(gate_f.mean().item()),
        "std": float(gate_f.std(unbiased=False).item()),
        "min": float(gate_f.min().item()),
        "max": float(gate_f.max().item()),
    }


def should_use_window_checkpointing(*, window_checkpointing: bool, training: bool, grad_enabled: bool) -> bool:
    """Return whether PMA window-level activation checkpointing should run."""
    return bool(window_checkpointing and training and grad_enabled)


def run_with_optional_window_checkpoint(
    layer_fn: Callable[[torch.Tensor], torch.Tensor], seq_t: torch.Tensor, *, enable_checkpoint: bool
) -> torch.Tensor:
    """Run a PMA window layer with optional activation checkpointing."""
    if enable_checkpoint:
        return checkpoint(layer_fn, seq_t, use_reentrant=False)
    return layer_fn(seq_t)


class GateModule(nn.Module):
    """Learn a mixing gate for recurrent state tensors.

    This module blends ``prev_states`` with ``rand_states`` according to a
    learnable gate ``g``.

    Default behavior (`use_residual=True`) uses the residual form::

        out = prev_states + g * (rand_states - prev_states)
            = (1 - g) * prev_states + g * rand_states

    With this default:
      - ``g -> 0`` preserves/copies ``prev_states``
      - ``g -> 1`` overwrites toward ``rand_states``

    Legacy non-residual behavior (`use_residual=False`) uses::

        out = g * prev_states + (1 - g) * rand_states

    which flips the 0/1 interpretation above.

    The ``gate_type`` argument controls the dimensionality of the gating
    parameters.

    Args:
        gate_type: One of ``"scalar"``, ``"dimwise"``, ``"exp_scalar"`` or
            ``"tokenwise"``.
        d_model: Embedding dimension ``D`` of the state tensors.
        n_states: Number of state tokens ``S`` per batch.
        name: Optional descriptive label used when logging gate statistics.
        temperature: Temperature divisor applied before the sigmoid.
        use_residual: Whether to use residual-form mixing (default: ``True``).
        init_logit: Optional initial logit for gate parameters.

    Returns:
        torch.Tensor: Weighted combination of ``prev_states`` and
        ``rand_states`` with shape ``[B, n_states, D]``.
    """

    def __init__(
        self,
        gate_type: str,
        d_model: int,
        n_states: int,
        name: str | None = None,
        temperature: float = 0.5,
        use_residual: bool = True,
        init_logit: float = None,
    ):
        super().__init__()
        self.gate_type = gate_type.lower()
        self.name = name  # Optional descriptive name for logging
        self.d_model = d_model
        self.n_states = n_states
        self.temperature = temperature  # Temperature for Gumbel-sigmoid
        self.use_residual = use_residual  # Whether to use residual formulation

        # Set default init_logit based on gate type if not provided
        if init_logit is None:
            init_logit = -2.0 if self.gate_type == "exp_scalar" else -0.5

        if self.gate_type == "scalar":
            # Single learnable scalar => shape [1]
            self.scalar_param = nn.Parameter(torch.full((1,), init_logit))
        elif self.gate_type == "dimwise":
            # Per-dimension gate => shape [d_model]
            self.dimwise_param = nn.Parameter(torch.full((d_model,), init_logit))
        elif self.gate_type == "exp_scalar":
            # β = exp(logit)
            self.scalar_param = nn.Parameter(torch.full((1,), init_logit))
        elif self.gate_type == "tokenwise":
            # Tokenwise gate applied per state token.
            self.tokenwise_linear = ZeroInitLinear(d_model, d_model, bias=True, bias_init=init_logit)
        else:
            raise ValueError(f"Unknown gate_type={gate_type}. Use 'scalar', 'dimwise', 'exp_scalar', or 'tokenwise'.")

        # Storage for the most recent gate statistics
        self.last_gate_stats = None
        self.supports_collect_stats = True

    def forward(
        self,
        prev_states: torch.Tensor,
        rand_states: torch.Tensor,
        *,
        collect_stats: bool = False,
        push_stats: bool = True,
    ) -> torch.Tensor:
        """Mix previous and random state tensors.

        Args:
            prev_states: Tensor of shape ``[B, n_states, D]`` containing the
                states carried over from the previous block.
            rand_states: Tensor of shape ``[B, n_states, D]`` with freshly
                initialised state tokens.

        Returns:
            torch.Tensor: Weighted combination of ``prev_states`` and
            ``rand_states`` with shape ``[B, n_states, D]``.
        """
        if self.gate_type == "scalar":
            # gate => [1] in [0,1] with temperature-controlled sigmoid
            gate = torch.sigmoid(self.scalar_param / self.temperature)
            # Use residual formulation if enabled for better gradient flow
            if self.use_residual:
                # Clamp inputs to avoid inf/nan propagation
                prev_states_safe = torch.where(torch.isinf(prev_states), torch.zeros_like(prev_states), prev_states)
                rand_states_safe = torch.where(torch.isinf(rand_states), torch.zeros_like(rand_states), rand_states)
                out = prev_states_safe + gate * (rand_states_safe - prev_states_safe)
            else:
                out = gate * prev_states + (1.0 - gate) * rand_states

            if collect_stats:
                with torch.no_grad():
                    self.last_gate_stats = _gate_stats_dict(gate)

        elif self.gate_type == "dimwise":
            # dimwise_param => [D] with temperature-controlled sigmoid
            gate_vec = torch.sigmoid(self.dimwise_param / self.temperature)  # => shape [D]
            # Use residual formulation if enabled for better gradient flow
            if self.use_residual:
                # Clamp inputs to avoid inf/nan propagation
                prev_states_safe = torch.where(torch.isinf(prev_states), torch.zeros_like(prev_states), prev_states)
                rand_states_safe = torch.where(torch.isinf(rand_states), torch.zeros_like(rand_states), rand_states)
                out = prev_states_safe + gate_vec * (rand_states_safe - prev_states_safe)
            else:
                out = gate_vec * prev_states + (1.0 - gate_vec) * rand_states

            if collect_stats:
                with torch.no_grad():
                    self.last_gate_stats = _gate_stats_dict(gate_vec)

        elif self.gate_type == "exp_scalar":
            # Apply temperature scaling to exp parameterization
            beta = torch.exp(self.scalar_param / self.temperature)  #   β ≥ 0
            gate = beta  # <-- make alias for final push

            # Use residual formulation if enabled for better gradient flow
            if self.use_residual:
                # Clamp inputs to avoid inf/nan propagation
                prev_states_safe = torch.where(torch.isinf(prev_states), torch.zeros_like(prev_states), prev_states)
                rand_states_safe = torch.where(torch.isinf(rand_states), torch.zeros_like(rand_states), rand_states)
                out = prev_states_safe + beta * (rand_states_safe - prev_states_safe)
            else:
                out = beta * prev_states + (1.0 - beta) * rand_states

            if collect_stats:
                with torch.no_grad():
                    self.last_gate_stats = _gate_stats_dict(beta)

        else:  # tokenwise
            gate_logits = self.tokenwise_linear(prev_states) / self.temperature
            gate_vals = torch.sigmoid(gate_logits)

            if self.use_residual:
                prev_states_safe = torch.where(torch.isinf(prev_states), torch.zeros_like(prev_states), prev_states)
                rand_states_safe = torch.where(torch.isinf(rand_states), torch.zeros_like(rand_states), rand_states)
                out = prev_states_safe + gate_vals * (rand_states_safe - prev_states_safe)
            else:
                out = gate_vals * prev_states + (1.0 - gate_vals) * rand_states

            if collect_stats:
                with torch.no_grad():
                    self.last_gate_stats = _gate_stats_dict(gate_vals)

        if not collect_stats:
            self.last_gate_stats = None

        # ── expose gate values to the thread-local buffer (if enabled) ─────
        if push_stats:
            gate_name = self.name
            if gate_name is None:
                if self.gate_type == "exp_scalar":
                    gate_name = "GateModule.exp_scalar"
                elif self.gate_type == "scalar":
                    gate_name = "GateModule.scalar"
                elif self.gate_type == "dimwise":
                    gate_name = "GateModule.dimwise"
                else:
                    gate_name = "GateModule.tokenwise"

            if self.gate_type == "exp_scalar":
                push_gate_stats(gate_name, gate)
            elif self.gate_type == "scalar":
                push_gate_stats(gate_name, gate)
            elif self.gate_type == "dimwise":
                push_gate_stats(gate_name, gate_vec)
            else:  # tokenwise
                push_gate_stats(gate_name, gate_vals)
        return out


def get_gate(
    gate_type: str,
    d_model: int,
    n_states: int,
    name: str | None = None,
    temperature: float = 0.5,
    use_residual: bool = True,
    init_logit: float = None,
):
    """
    Returns a GateModule for the specified gate_type.
      'scalar', 'dimwise', 'exp_scalar', or 'tokenwise'.

    Args:
        gate_type: Type of gate ('scalar', 'dimwise', 'exp_scalar', 'tokenwise')
        d_model: Model dimension
        n_states: Number of states
        name: Optional name for logging
        temperature: Temperature for Gumbel-sigmoid activation
        use_residual: Whether to use residual mixing. If True, gate values
            follow the copy->overwrite direction (0 preserves prev, 1 moves
            toward rand).
        init_logit: Initial logit value for gate parameters
    """
    return GateModule(
        gate_type=gate_type,
        d_model=d_model,
        n_states=n_states,
        name=name,
        temperature=temperature,
        use_residual=use_residual,
        init_logit=init_logit,
    )


class OverlapSelectorMostContext(nn.Module):
    """Deterministic resolver for stride<window overlap: choose the earliest
    window that contains each token (max valid past in causal masks).

    Supports two input formats:
      - forward_from_windows(window_emb, meta):
          window_emb: [B, num_win, W, D]
          meta: {'front_pad','L','window_size','window_stride','num_win'}
      - forward_from_overlaps(x_4d, patch_mask):
          x_4d: [B, L, O, D], patch_mask: [B, L, O] or None
    """

    def __init__(self, name: str | None = None):
        super().__init__()
        self.name = name or "OverlapSelectorMostContext"

    @torch.no_grad()
    def _indices(self, *, L: int, front_pad: int, W: int, stride: int, num_win: int, device):
        p = torch.arange(front_pad, front_pad + L, device=device)
        w_star = torch.div(p - (W - 1) + stride - 1, stride, rounding_mode='floor')
        w_star = w_star.clamp_(0, num_win - 1)
        u_star = p - w_star * stride
        return w_star.long(), u_star.long()

    def forward_from_windows(self, window_emb: torch.Tensor, *, meta: Dict) -> torch.Tensor:
        B, num_win, W, D = window_emb.shape
        w_star, u_star = self._indices(
            L=int(meta['L']),
            front_pad=int(meta['front_pad']),
            W=W,
            stride=int(meta['window_stride']),
            num_win=num_win,
            device=window_emb.device,
        )
        L = w_star.numel()
        b_idx = torch.arange(B, device=window_emb.device)[:, None].expand(B, L)
        w_idx = w_star.view(1, L).expand(B, L)
        u_idx = u_star.view(1, L).expand(B, L)
        out = window_emb[b_idx, w_idx, u_idx, :]
        return out

    def forward_from_overlaps(self, x_4d: torch.Tensor, *, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pick earliest valid along O. If no mask is provided, choose index 0.
        Args:
            x_4d: [B, L, O, D]
            patch_mask: [B, L, O] boolean or None
        Returns: [B, L, D]
        """
        B, L, overlap_count, D = x_4d.shape
        if patch_mask is None:
            idx = torch.zeros(B, L, dtype=torch.long, device=x_4d.device)
        else:
            if patch_mask.dim() == 2:
                valid = patch_mask.unsqueeze(-1).expand(-1, -1, overlap_count)
            else:
                valid = patch_mask
            idx = valid.to(x_4d.dtype).argmax(dim=-1)  # first True (1) or 0 if all False
        gathered = torch.gather(x_4d, 2, idx[..., None, None].expand(B, L, 1, D))
        return gathered.squeeze(2)


class LocalOffsetEmbedding(nn.Module):
    """
    A small embedding that maps each local offset j in [0..window_size-1]
    to a d_model-dimensional vector, then scales it down by `scale_factor`.
    """

    def __init__(self, window_size: int, d_model: int, scale_factor: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        self.scale_factor = scale_factor

        self.embedding = nn.Embedding(num_embeddings=window_size, embedding_dim=d_model)

        # Optional initialization of embedding weights, e.g. xavier:
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, offsets: torch.Tensor) -> torch.Tensor:
        """
        offsets: LongTensor [B, window_size] with values in [0..window_size-1].
        Returns [B, window_size, d_model] scaled by scale_factor.
        """
        emb = self.embedding(offsets)  # [B, window_size, d_model]
        return emb * self.scale_factor


class ChunkOffsetEmbedding(nn.Module):
    """
    If you want each chunk i to have a 'global offset' embedded in the state tokens
    (rather than all patches), you can add that in. For example, chunk i => embed(i).
    Or chunk i => embed(i * stride).
    """

    def __init__(self, max_chunks: int, d_model: int, scale_factor: float = 1.0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=max_chunks, embedding_dim=d_model)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.scale_factor = scale_factor

    def forward(self, chunk_idx: int, device: torch.device) -> torch.Tensor:
        """
        chunk_idx: integer index for the chunk i in [0..max_chunks-1].
        Returns a single [d_model] embedding scaled by scale_factor.
        """
        idx = torch.tensor([chunk_idx], device=device, dtype=torch.long)
        emb = self.embedding(idx)  # [1, d_model]
        return emb * self.scale_factor


class SkipGate(nn.Module):
    """Fuse a residual path with aggregated features using a learnable gate.

    ``SkipGate`` controls how much of the aggregated feature is mixed into the
    residual signal.  The ``kind`` argument selects the parameterisation of the
    gating weights.

    Args:
        d_model: Feature dimension ``D`` of the inputs.
        kind: One of ``"scalar"``, ``"dimwise"``, ``"exp_scalar"`` or
            ``"tokenwise"``.
        temp: Logit temperature. Use ``1.0`` to disable scaling. Values
            ``>1`` soften the gate (toward 0.5) while values ``<1`` sharpen it.
        name: Optional descriptive label used when logging gate statistics.

    Returns:
        torch.Tensor: Output tensor ``skip + gate * agg`` with the same shape as
        ``skip`` and ``agg``.
    """

    def __init__(
        self, d_model: int, kind: str = "scalar", temp: float = 1.0, init_logit: float = -2.0, name: str | None = None
    ) -> None:
        super().__init__()
        self.name = name  # Optional descriptive name for logging
        raw_kind = kind.lower()
        tokenwise_init_mode = "zero_bias"
        if raw_kind in {"tokenwise_legacy", "tokenwise_legacy_init", "tokenwise_legacy_constant_weight"}:
            kind = "tokenwise"
            tokenwise_init_mode = "legacy_constant_weight"
        else:
            kind = raw_kind
        self.kind = kind  # Store kind
        self.config_kind = raw_kind
        self.tokenwise_init_mode = tokenwise_init_mode
        self.temp = float(temp)
        self.supports_collect_stats = True
        if kind == "scalar":
            # Initialize with configurable logit value
            self.param = nn.Parameter(torch.full((1,), init_logit))
        elif kind == "dimwise":
            # Initialize with configurable logit value
            self.param = nn.Parameter(torch.full((d_model,), init_logit))
        elif kind == "exp_scalar":
            self.param = nn.Parameter(torch.full((1,), init_logit))  # Use configurable init
        elif kind == "tokenwise":
            if tokenwise_init_mode == "legacy_constant_weight":
                # Diagnostic compatibility mode for pre-678aad83 tokenwise skip
                # gates. This intentionally consumes RNG through nn.Linear
                # construction before overwriting the weights, matching the old
                # initialization side effect on later module parameters.
                self.param = nn.Linear(d_model, d_model, bias=False)
                nn.init.constant_(self.param.weight, float(init_logit) / math.sqrt(d_model))
            else:
                # Tokenwise gate starts equivalent to scalar/dimwise at init:
                # zero weights + bias(init_logit) => sigmoid(init_logit) everywhere.
                self.param = ZeroInitLinear(d_model, d_model, bias=True, bias_init=init_logit)
        else:
            raise ValueError(
                "kind must be scalar | dimwise | exp_scalar | tokenwise " "| tokenwise_legacy_constant_weight"
            )

        # Storage for the most recent gate statistics
        self.last_gate_stats = None

    def forward(self, skip: torch.Tensor, agg: torch.Tensor, *, collect_stats: bool = False) -> torch.Tensor:
        """Blend skip and aggregated features.

        Args:
            skip: Residual tensor of shape ``[B, L, D]``.
            agg: Aggregated feature tensor of the same shape as ``skip``.

        Returns:
            torch.Tensor: Fused tensor with shape ``[B, L, D]``.
        """
        if self.kind == "tokenwise":
            gate_logits = self.param(skip) / self.temp
            gate = torch.sigmoid(gate_logits)
        elif self.kind in ("scalar", "dimwise"):
            # Ensure param has correct shape for broadcasting
            view_shape = [1] * (skip.ndim - 1) + [-1]  # Make view shape match skip's ndim
            gate_logits = self.param.view(view_shape) / self.temp
            gate = torch.sigmoid(gate_logits)
        else:  # exp_scalar
            view_shape = [1] * (skip.ndim - 1) + [-1]
            # Apply temperature scaling to exp parameterization
            gate = torch.exp(self.param.view(view_shape) / self.temp)  # β ≥ 0   (unbounded, lets network scale)

        if collect_stats:
            with torch.no_grad():
                self.last_gate_stats = _gate_stats_dict(gate)
        else:
            self.last_gate_stats = None

        # ── expose gate values to the thread-local buffer (if enabled) ─────
        gate_name = self.name or f"SkipGate.{self.kind}"
        push_gate_stats(gate_name, gate)

        # Original skip + gated aggregated feature
        return skip + gate * agg


def apply_skip_gate_warmup(model: nn.Module, epoch: int, *, init_logit: float, warm_epochs: int) -> None:
    """Linearly anneal skip gate logits during warmup.

    Keeps gates close to identity for the first warm_epochs,
    then allows free learning.

    Args:
        model: Model containing skip gates
        epoch: Current epoch (1-indexed)
        init_logit: Initial logit value
        warm_epochs: Number of warmup epochs
    """
    if warm_epochs <= 0 or epoch > warm_epochs:
        return

    factor = 1.0 - epoch / float(warm_epochs)
    new_logit = init_logit * factor

    for m in model.modules():
        if isinstance(m, SkipGate):
            # Handle SkipGate - param can be either Parameter or Linear module
            if hasattr(m, "param"):
                if isinstance(m.param, nn.Linear):
                    # Tokenwise case - param is a Linear module
                    if getattr(m, "tokenwise_init_mode", "zero_bias") == "legacy_constant_weight":
                        d_model = m.param.weight.shape[1]
                        m.param.weight.data.fill_(new_logit / math.sqrt(d_model))
                    else:
                        m.param.weight.data.zero_()
                        if m.param.bias is not None:
                            m.param.bias.data.fill_(new_logit)
                else:
                    # Scalar/dimwise/exp_scalar cases - param is a Parameter
                    m.param.data.fill_(new_logit)
        elif isinstance(m, GateModule):
            # Handle GateModule - different attributes for different gate types
            if hasattr(m, "scalar_param"):  # scalar or exp_scalar
                m.scalar_param.data.fill_(new_logit)
            elif hasattr(m, "dimwise_param"):  # dimwise
                m.dimwise_param.data.fill_(new_logit)
            elif hasattr(m, "tokenwise_linear"):  # tokenwise
                m.tokenwise_linear.weight.data.zero_()
                if m.tokenwise_linear.bias is not None:
                    m.tokenwise_linear.bias.data.fill_(new_logit)


class OverlapAggregator(nn.Module):
    """
    Multi-head attention aggregator for merging [B, L, O, D] => [B, L, D].
    If O=1, it just returns that single embedding per patch.

    By default, query = the mean of O embeddings, but you can alter that if desired.
    """

    def __init__(self, d_model, num_heads=1, dropout=0.1, debug=False):
        super().__init__()
        from trd.hosts.pmt.transformer import MultiHeadAttention

        self.debug = debug
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, O, D]
        Returns: [B, L, D]
        """
        B, L, overlap_count, D = x.shape
        if self.debug:
            print(f"[OverlapAggregator] Input shape: {x.shape}")

        if overlap_count == 1:
            x_out = x.squeeze(2)  # [B,L,D]
            if self.debug:
                print("[OverlapAggregator] No overlap => direct return.")
            return x_out

        # Flatten across B*L
        x_agg = x.view(B * L, overlap_count, D)  # [B*L, O, D]

        # Query = mean of the O embeddings
        Q = x_agg.mean(dim=1, keepdim=True)  # [B*L,1,D]
        K = x_agg
        V = x_agg

        attn_out = self.attention(Q, K, V)  # => [B*L,1,D]
        x_out = attn_out.view(B, L, D)

        if self.debug:
            print(f"[OverlapAggregator] Output shape: {x_out.shape}")
        return x_out
