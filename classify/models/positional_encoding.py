"""Sinusoidal temporal positional embeddings.

论文: sinusoidal positional embeddings added along the temporal axis.
约束: 不允许用可学习 positional embedding 替代; T 可配置, 默认 8.

输出: shape [T, D], 与 x ([B, T, D]) 相加.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """标准 Transformer sinusoidal PE (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    输入  x: [B, T, D]
    输出  x + pe: [B, T, D]
    pe (buffer): [T, D]
    """

    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        assert d_model % 2 == 0, f"d_model must be even for sinusoidal PE, got {d_model}"
        pe = torch.zeros(max_len, d_model)  # [max_len, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # [D/2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, D], 不可学习
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] -> x + pe[:T]"""
        T = x.size(1)
        assert T <= self.max_len, f"sequence length {T} exceeds max_len {self.max_len}"
        return x + self.pe[:T].unsqueeze(0).to(dtype=x.dtype, device=x.device)
