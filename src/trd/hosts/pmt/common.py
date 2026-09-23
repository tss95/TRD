import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class PortableConv1d(nn.Conv1d):
    def __init__(
        self,
        in_ch,
        out_ch,
        k: int | None = None,
        *,
        kernel_size: int | None = None,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        **kw,
    ):
        k = kernel_size if k is None else k
        if k is None:
            raise TypeError("PortableConv1d missing required kernel size (positional `k` or keyword `kernel_size`)")
        if padding is None:
            # Preserve the historical k//2 behavior at dilation=1, including
            # the extra right-edge output produced by even kernels. For odd
            # kernels this is exact symmetric "same" padding at any dilation.
            padding = (int(dilation) * (int(k) - 1) + 1) // 2
        super().__init__(
            in_ch, out_ch, k, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, **kw
        )
        self._fast_path: bool | None = None

    def _use_unfold(self) -> bool:
        if self._fast_path is not None:
            return self._fast_path
        if not torch.cuda.is_available():
            self._fast_path = False
            return False
        if self.dilation[0] != 1 or self.groups != 1:
            self._fast_path = False
            return False
        name = torch.cuda.get_device_name(0).lower()
        is_amd = any(t in name for t in ("amd", "instinct", "mi"))
        hip = getattr(torch.version, "hip", "")
        self._fast_path = (is_amd and hip.startswith("6.0")) or bool(os.getenv("FORCE_UNFOLD"))
        return self._fast_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_unfold():
            return F.conv1d(
                x,
                self.weight,
                self.bias,
                stride=self.stride[0],
                padding=self.padding[0],
                dilation=self.dilation[0],
                groups=self.groups,
            )
        B, C, L = x.shape
        k = self.kernel_size[0]
        s = self.stride[0]
        patches = F.unfold(x.unsqueeze(-1), (k, 1), stride=(s, 1), padding=(self.padding[0], 0))
        if patches.numel() > 2e8:
            raise RuntimeError(
                f"Unfold buffer too large: {patches.numel()} elements. Consider using standard conv1d path."
            )
        w2d = self.weight.view(self.out_channels, -1).t()
        if w2d.size(0) != patches.size(1):
            return F.conv1d(
                x,
                self.weight,
                self.bias,
                stride=self.stride[0],
                padding=self.padding[0],
                dilation=self.dilation[0],
                groups=self.groups,
            )
        y = torch.matmul(patches.transpose(1, 2), w2d)
        if self.bias is not None:
            y += self.bias.view(1, 1, -1)
        return y.permute(0, 2, 1)


def is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_bn_layer(num_features, eps=1e-8):
    if is_distributed():
        return nn.SyncBatchNorm(num_features, eps=eps)
    return nn.BatchNorm1d(num_features, eps=eps)


def swish(x):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim, init_xavier=True):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)
        self.V = nn.Linear(in_dim, out_dim, bias=True)
        if init_xavier:
            nn.init.xavier_uniform_(self.W.weight)
            nn.init.xavier_uniform_(self.V.weight)
            if self.W.bias is not None:
                nn.init.zeros_(self.W.bias)
            if self.V.bias is not None:
                nn.init.zeros_(self.V.bias)

    def forward(self, x):
        w_out = self.W(x)
        w_swish = swish(w_out)
        v_out = self.V(x)
        return w_swish * v_out


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with on-demand support for long sequences."""

    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.d_model = int(d_model)
        self.register_buffer('pe', self._build_pe(int(max_len), device=None).unsqueeze(0))

    def _build_pe(self, seq_length: int, *, device: torch.device | None) -> torch.Tensor:
        pe = torch.zeros(seq_length, self.d_model, device=device)
        position = torch.arange(0, seq_length, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe

    def forward(self, x):
        seq_length = x.size(1)
        if seq_length <= self.pe.size(1):
            pe = self.pe[:, :seq_length, :].to(device=x.device)
        else:
            pe = self._build_pe(seq_length, device=x.device).unsqueeze(0)
        return x + pe
