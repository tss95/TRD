import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from trd.hosts.pmt.common import PortableConv1d, PositionalEncoding, get_bn_layer, SwiGLU


class InputBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        conv_num_kernels: int = 64,
        conv_kernel_size: int = 3,
        conv_stride: int = 1,
        conv_dropout: float = 0.1,
        activation: str = "relu",
        use_conv: bool = True,
        n_conv_layers: int = 1,
        debug: bool = False,
        name: str = "InputBlock",
        auto_transpose: bool = False,
        cfg=None,
        add_cls: bool = True,
        apply_token_pe: bool = False,
        token_pe_skip_cls: bool = True,
        dilated_context_dilations: Sequence[int] | None = None,
        dilated_context_kernel_size: int = 3,
        dilated_context_residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.debug = debug
        self.name = name
        self.use_conv = use_conv
        self.d_model = d_model
        self.activation = activation.lower()
        self.auto_transpose = auto_transpose
        self.cfg = cfg
        self.add_cls = bool(add_cls)
        self.token_pe_skip_cls = bool(token_pe_skip_cls)
        self.dilated_context_dilations = self._validate_dilated_context_dilations(dilated_context_dilations)
        if isinstance(dilated_context_kernel_size, bool) or not isinstance(dilated_context_kernel_size, int):
            raise ValueError(
                "input_block_dilated_kernel_size must be an odd positive integer, "
                f"got {dilated_context_kernel_size!r}"
            )
        self.dilated_context_kernel_size = int(dilated_context_kernel_size)
        if self.dilated_context_kernel_size <= 0 or self.dilated_context_kernel_size % 2 == 0:
            raise ValueError(
                "input_block_dilated_kernel_size must be an odd positive integer, "
                f"got {self.dilated_context_kernel_size}"
            )
        try:
            self.dilated_context_residual_scale = float(dilated_context_residual_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "input_block_dilated_residual_scale must be a finite positive number, "
                f"got {dilated_context_residual_scale!r}"
            ) from exc
        if not math.isfinite(self.dilated_context_residual_scale) or self.dilated_context_residual_scale <= 0:
            raise ValueError(
                "input_block_dilated_residual_scale must be a finite positive number, "
                f"got {dilated_context_residual_scale!r}"
            )
        if self.dilated_context_dilations and not self.use_conv:
            raise ValueError("dilated input context requires use_conv=True")
        if isinstance(n_conv_layers, bool):
            raise ValueError("input_block_n_conv_layers must be an integer >= 1, not bool")
        try:
            self.n_conv_layers = int(n_conv_layers)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"input_block_n_conv_layers must be an integer >= 1, got {n_conv_layers!r}") from exc
        if self.n_conv_layers < 1:
            raise ValueError(f"input_block_n_conv_layers must be >= 1, got {self.n_conv_layers}")

        self.expected_channels = in_channels
        if cfg is None and auto_transpose:
            raise ValueError("cfg is required when auto_transpose=True")
        self.expected_timesteps = cfg.data_parameters.seq_len_inferred if cfg is not None else None

        # Determine normalization type from config (default to layernorm if not specified)
        norm_type = "layernorm"  # Default to layernorm for safety
        if cfg is not None and hasattr(cfg, "model") and hasattr(cfg.model, "input_block_norm_type"):
            norm_type = cfg.model.input_block_norm_type.lower()

        # Historical field name; currently interpreted as GroupNorm num_groups.
        group_size = 16
        if cfg is not None and hasattr(cfg, "model") and hasattr(cfg.model, "input_block_group_size"):
            group_size = cfg.model.input_block_group_size

        # Helper function to create normalization layer
        def get_norm_layer(num_features, norm_type="layernorm", num_groups=16):
            if norm_type == "batchnorm":
                return get_bn_layer(num_features, eps=1e-8)
            elif norm_type == "layernorm":
                # LayerNorm expects input shape (B, L, C) so we'll handle dimension in forward
                return nn.LayerNorm(num_features, eps=1e-8)
            elif norm_type == "groupnorm":
                # Ensure num_groups divides num_features evenly
                while num_features % num_groups != 0 and num_groups > 1:
                    num_groups = num_groups // 2
                if num_groups == 0:
                    num_groups = 1
                return nn.GroupNorm(num_groups, num_features, eps=1e-8)
            elif norm_type == "none":
                # Explicit opt-out used by controlled tokenizer-geometry audits.
                # Keep this distinct from an unknown string so configuration
                # typos still fail hard below.
                return nn.Identity()
            else:
                raise ValueError(f"Unknown norm type: {norm_type}")

        self.norm_type = norm_type

        if self.use_conv:
            self.conv1d = PortableConv1d(
                in_channels, conv_num_kernels, kernel_size=conv_kernel_size, stride=conv_stride, bias=False
            )
            self.norm = get_norm_layer(conv_num_kernels, norm_type, num_groups=group_size)
            self.extra_conv_layers = nn.ModuleList(
                [
                    PortableConv1d(
                        conv_num_kernels, conv_num_kernels, kernel_size=conv_kernel_size, stride=1, bias=False
                    )
                    for _ in range(self.n_conv_layers - 1)
                ]
            )
            self.extra_norms = nn.ModuleList(
                [get_norm_layer(conv_num_kernels, norm_type, num_groups=group_size) for _ in self.extra_conv_layers]
            )
            # Avoid in-place activations to prevent autograd issues
            self.act1 = nn.ReLU() if self.activation in ["relu", "swiglu"] else nn.GELU()
            self.dropout = nn.Dropout(conv_dropout)
            self.num_kernels = conv_num_kernels
        else:
            self.linear_in = nn.Linear(in_channels, conv_num_kernels, bias=False)
            self.norm = get_norm_layer(conv_num_kernels, norm_type, num_groups=group_size)
            # Avoid in-place activations to prevent autograd issues
            self.act1 = nn.ReLU() if self.activation in ["relu", "swiglu"] else nn.GELU()
            self.dropout = nn.Dropout(conv_dropout)
            self.num_kernels = conv_num_kernels
            self.extra_conv_layers = nn.ModuleList()
            self.extra_norms = nn.ModuleList()

        if self.activation in ["relu", "gelu"]:
            self.projection = nn.Linear(self.num_kernels, d_model, bias=True)
            # Avoid in-place activations to prevent autograd issues
            self.act2 = nn.ReLU() if self.activation == "relu" else nn.GELU()
            self.use_swiglu = False
        elif self.activation == "swiglu":
            self.swiGLU = SwiGLU(self.num_kernels, d_model, init_xavier=True)
            self.use_swiglu = True
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        if self.add_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.num_kernels) * 0.02)
        self.token_pe = PositionalEncoding(d_model=self.d_model) if bool(apply_token_pe) else None
        self._initialize_weights()

        # Construct the opt-in context stack only after initializing the
        # incumbent stem. This keeps all shared standard-stem parameters bitwise
        # paired at a fixed seed; the builder separately preserves downstream
        # RNG state for controlled ablations.
        self.dilated_context_convs = nn.ModuleList(
            [
                PortableConv1d(
                    self.num_kernels,
                    self.num_kernels,
                    kernel_size=self.dilated_context_kernel_size,
                    stride=1,
                    dilation=dilation,
                    groups=self.num_kernels,
                    bias=False,
                )
                for dilation in self.dilated_context_dilations
            ]
        )
        for conv in self.dilated_context_convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="linear")

    @staticmethod
    def _validate_dilated_context_dilations(dilations: Sequence[int] | None) -> tuple[int, ...]:
        if dilations is None:
            return ()
        if isinstance(dilations, (str, bytes)) or not isinstance(dilations, Sequence):
            raise ValueError(
                "input_block_dilations must be a non-empty sequence of positive integers, " f"got {dilations!r}"
            )
        values = tuple(dilations)
        if not values:
            raise ValueError("input_block_dilations must contain at least one positive integer")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("input_block_dilations must contain only positive integers, " f"got {list(values)!r}")
        return values

    def _initialize_weights(self):
        if self.use_conv:
            for conv in [self.conv1d, *self.extra_conv_layers]:
                nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)
        else:
            nn.init.kaiming_normal_(self.linear_in.weight, mode='fan_out', nonlinearity='relu')

        for norm in [self.norm, *self.extra_norms]:
            if hasattr(norm, 'weight'):
                nn.init.constant_(norm.weight, 1.0)
            if hasattr(norm, 'bias'):
                nn.init.constant_(norm.bias, 0.0)

    @staticmethod
    def _match_sequence_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
        current_length = x.size(-1)
        if current_length == target_length:
            return x
        if current_length > target_length:
            return x[..., :target_length]
        pad_width = target_length - current_length
        return torch.nn.functional.pad(x, (0, pad_width))

    @staticmethod
    def _match_mask_length(mask: torch.Tensor, target_length: int) -> torch.Tensor:
        current_length = mask.size(-1)
        if current_length == target_length:
            return mask
        if current_length > target_length:
            return mask[..., :target_length]
        pad = mask.new_zeros((*mask.shape[:-1], target_length - current_length))
        return torch.cat([mask, pad], dim=-1)

    @classmethod
    def _pool_valid_mask(
        cls, mask: torch.Tensor, *, kernel_size: int, stride: int, padding: int, target_length: int
    ) -> torch.Tensor:
        mask_f = mask.unsqueeze(1).float()
        pooled_valid = torch.nn.functional.avg_pool1d(mask_f, kernel_size, stride, padding=padding, ceil_mode=False)
        pooled_coverage = torch.nn.functional.avg_pool1d(
            torch.ones_like(mask_f), kernel_size, stride, padding=padding, ceil_mode=False
        )
        valid_fraction = pooled_valid / pooled_coverage.clamp_min(1e-6)
        pooled_mask = valid_fraction.squeeze(1) > 0.5
        return cls._match_mask_length(pooled_mask, target_length)

    def _apply_norm(self, x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        if self.norm_type == "layernorm":
            x = x.transpose(1, 2)
            x = norm(x)
            return x.transpose(1, 2)
        if self.norm_type in ["batchnorm", "groupnorm"]:
            return norm(x)
        return norm(x)

    def _apply_dilated_context(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Apply residual context without allowing invalid patch features to leak."""

        for conv in self.dilated_context_convs:
            target_length = x.size(-1)
            context_input = x.masked_fill(~valid_mask.unsqueeze(1), 0.0)
            context_update = conv(context_input)
            context_update = self._match_sequence_length(context_update, target_length)
            x = x + self.dilated_context_residual_scale * context_update
        return x

    def _check_and_fix_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        if not self.auto_transpose:
            return x
        if x.dim() != 3:
            return x
        B, dim1, dim2 = x.shape
        needs = False
        if self.use_conv:
            if self.expected_channels > 0:
                if dim1 > 5 * self.expected_channels and dim2 == self.expected_channels:
                    needs = True
            else:
                if dim1 > dim2 * 4:
                    needs = True
        else:
            if self.expected_channels > 0 and self.expected_timesteps:
                if dim1 == self.expected_channels and dim2 >= self.expected_timesteps * 0.9:
                    needs = True
            elif dim1 < dim2 / 5.0:
                needs = True
            elif self.expected_channels > 0 and dim1 == self.expected_channels and dim2 > dim1 * 2:
                needs = True
        return x.transpose(1, 2) if needs else x

    def _project_output_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_swiglu:
            tokens = self.projection(tokens)
            return self.act2(tokens)
        return self.swiGLU(tokens)

    def forward(self, x: torch.Tensor, *, return_mask: bool = True):
        if self.debug:
            print(f"\n[{self.name}] Input shape: {x.shape}")

        x = self._check_and_fix_input_shape(x)
        if self.use_conv:
            raw_mask = ~torch.isnan(x).any(dim=1)
        else:
            mask_ts = ~torch.isnan(x).any(dim=-1)

        x = torch.nan_to_num(x, nan=0.0)

        if self.use_conv:
            assert self.conv1d.stride[0] > 0, "conv1d stride must be ≥ 1"
            x = self.conv1d(x)
            mask_ts = self._pool_valid_mask(
                raw_mask,
                kernel_size=self.conv1d.kernel_size[0],
                stride=self.conv1d.stride[0],
                padding=self.conv1d.padding[0],
                target_length=x.size(-1),
            )
            x = self._apply_norm(x, self.norm)
            x = self.act1(x)
            x = self.dropout(x)
            x = self._apply_dilated_context(x, mask_ts)
            for conv, norm in zip(self.extra_conv_layers, self.extra_norms):
                target_length = x.size(-1)
                x = conv(x)
                x = self._match_sequence_length(x, target_length)
                mask_ts = self._pool_valid_mask(
                    mask_ts,
                    kernel_size=conv.kernel_size[0],
                    stride=conv.stride[0],
                    padding=conv.padding[0],
                    target_length=target_length,
                )
                x = self._apply_norm(x, norm)
                x = self.act1(x)
                x = self.dropout(x)
        else:
            x = self.linear_in(x)
            x = x.transpose(1, 2)
            x = self._apply_norm(x, self.norm)
            x = self.act1(x)
            x = self.dropout(x)

        x = x.transpose(1, 2)

        if self.debug:
            print(f"[{self.name}] After conv_block: {x.shape}")

        if self.token_pe is not None:
            x = self._project_output_tokens(x)
            if self.add_cls:
                B = x.shape[0]
                cls_tok = self.cls_token.expand(B, -1, -1)
                cls_tok = self._project_output_tokens(cls_tok)
                if self.token_pe_skip_cls:
                    x = self.token_pe(x)
                    x = torch.cat((cls_tok, x), dim=1)
                else:
                    x = torch.cat((cls_tok, x), dim=1)
                    x = self.token_pe(x)
                mask_ts = torch.cat([mask_ts.new_ones(B, 1), mask_ts], dim=1)
            else:
                x = self.token_pe(x)

            if self.debug:
                print(f"[{self.name}] Output shape: {x.shape},  mask valid={mask_ts.sum().item()}/{mask_ts.numel()}")

            return (x, mask_ts) if return_mask else x

        if self.add_cls:
            B = x.shape[0]
            cls_tok = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tok, x), dim=1)
            mask_ts = torch.cat([mask_ts.new_ones(B, 1), mask_ts], dim=1)

        x = self._project_output_tokens(x)

        if self.debug:
            print(f"[{self.name}] Output shape: {x.shape},  mask valid={mask_ts.sum().item()}/{mask_ts.numel()}")

        return (x, mask_ts) if return_mask else x
