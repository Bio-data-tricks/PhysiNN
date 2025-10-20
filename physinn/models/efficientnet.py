"""1D EfficientNet-style encoder components."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


def _make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    min_value = min_value or divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def Norm1d(channels: int) -> nn.GroupNorm:
    groups = 8 if channels >= 8 else 1
    return nn.GroupNorm(groups, channels)


class ConvBNAct1d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        bias: bool = False,
        act: bool = True,
    ) -> None:
        padding = padding if padding is not None else kernel_size // 2
        modules = [nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias), Norm1d(out_channels)]
        if act:
            modules.append(SiLU())
        super().__init__(*modules)


class SE1d(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.25) -> None:
        super().__init__()
        hidden = max(1, int(channels * se_ratio))
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, hidden, 1)
        self.act = SiLU()
        self.fc2 = nn.Conv1d(hidden, channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        out = self.avg(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        return x * self.gate(out)


class FusedMBConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        s: int,
        expand_ratio: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        mid_channels = _make_divisible(in_channels * expand_ratio, 8)
        self.expand_conv = ConvBNAct1d(in_channels, mid_channels, k, s)
        self.project_conv = ConvBNAct1d(mid_channels, out_channels, 1, 1, act=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.use_residual = s == 1 and in_channels == out_channels

    def forward(self, x):
        residual = x
        x = self.expand_conv(x)
        x = self.project_conv(x)
        if self.use_residual:
            x = residual + self.drop_path(x)
        return x


class MBConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        s: int,
        expand_ratio: float,
        se_ratio: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        mid_channels = _make_divisible(in_channels * expand_ratio, 8)
        self.expand = in_channels != mid_channels
        if self.expand:
            self.expand_conv = ConvBNAct1d(in_channels, mid_channels, 1, 1)
        self.depthwise = ConvBNAct1d(mid_channels, mid_channels, k, s, groups=mid_channels)
        self.se = SE1d(mid_channels, se_ratio=se_ratio)
        self.project = ConvBNAct1d(mid_channels, out_channels, 1, 1, act=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.use_residual = s == 1 and in_channels == out_channels

    def forward(self, x):
        residual = x
        if self.expand:
            x = self.expand_conv(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_residual:
            x = residual + self.drop_path(x)
        return x


_EFFV2_CFGS: Dict[str, List[Tuple[str, int, int, int, int, float]]] = {
    "s": [
        ("fused", 2, 24, 3, 1, 1.0),
        ("fused", 4, 48, 3, 2, 4.0),
        ("fused", 4, 64, 3, 2, 4.0),
        ("mb", 6, 128, 3, 2, 4.0),
        ("mb", 9, 160, 3, 1, 6.0),
        ("mb", 15, 256, 3, 2, 6.0),
    ],
    "m": [
        ("fused", 3, 24, 3, 1, 1.0),
        ("fused", 5, 48, 3, 2, 4.0),
        ("fused", 5, 80, 3, 2, 4.0),
        ("mb", 7, 160, 3, 2, 6.0),
        ("mb", 11, 176, 3, 1, 6.0),
        ("mb", 15, 304, 3, 2, 6.0),
    ],
    "l": [
        ("fused", 4, 32, 3, 1, 1.0),
        ("fused", 7, 64, 3, 2, 4.0),
        ("fused", 7, 96, 3, 2, 4.0),
        ("mb", 7, 192, 3, 2, 6.0),
        ("mb", 10, 224, 3, 1, 6.0),
        ("mb", 14, 384, 3, 2, 6.0),
    ],
}


class EfficientNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        variant: str = "s",
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.1,
        stem_channels: int | None = None,
    ) -> None:
        super().__init__()
        cfg = _EFFV2_CFGS[variant]
        stem_out = stem_channels or 32
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_out, kernel_size=3, stride=2, padding=1, bias=False),
            Norm1d(stem_out),
            SiLU(),
        )
        in_c = stem_out
        blocks = []
        total_blocks = sum(int(math.ceil(r * depth_mult)) for (_, r, *_ ) in cfg)
        block_idx = 0
        for kind, repeats, out_c, kernel, stride, expand in cfg:
            out_c = _make_divisible(out_c * width_mult, 8)
            repeats = int(math.ceil(repeats * depth_mult))
            for repeat_idx in range(repeats):
                stride_value = stride if repeat_idx == 0 else 1
                drop = drop_path_rate * block_idx / max(1, total_blocks - 1)
                if kind == "fused":
                    block = FusedMBConv1d(in_c, out_c, k=kernel, s=stride_value, expand_ratio=expand, drop_path=drop)
                else:
                    block = MBConv1d(
                        in_c,
                        out_c,
                        k=kernel,
                        s=stride_value,
                        expand_ratio=expand,
                        se_ratio=se_ratio,
                        drop_path=drop,
                    )
                blocks.append(block)
                in_c = out_c
                block_idx += 1
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Identity()
        self.feat_dim = in_c

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x, None

