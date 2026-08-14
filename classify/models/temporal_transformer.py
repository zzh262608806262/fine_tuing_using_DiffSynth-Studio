"""Temporal Transformer encoder (4 layers, 8 heads, GELU, dropout 0.1).

仅建模 T=8 帧之间的时间关系 (不建模 patch 关系).
输入: [B, T, D]  (+ sinusoidal temporal PE 后)
输出: [B, T, D]
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .positional_encoding import SinusoidalPositionalEncoding


class TemporalTransformerEncoder(nn.Module):
    """标准 Transformer encoder + sinusoidal temporal PE.

    输入  x: [B, T, D]
    输出  y: [B, T, D]
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        activation: str = "gelu",
        dropout: float = 0.1,
        pos_encoding: str = "sinusoidal",
        max_len: int = 1024,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim {hidden_dim} 必须能被 num_heads {num_heads} 整除"
        )
        assert pos_encoding == "sinusoidal", "论文要求 sinusoidal, 不允许可学习 PE"

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * mlp_ratio,
            dropout=dropout,
            activation=activation,
            batch_first=True,   # [B, T, D]
            norm_first=True,    # Pre-LN, 训练更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, max_len=max_len)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] -> [B, T, D]"""
        x = self.pos_encoding(x)        # + sinusoidal temporal PE, [B, T, D]
        x = self.encoder(x)             # [B, T, D]
        x = self.norm(x)                # 最终 LayerNorm
        return x
