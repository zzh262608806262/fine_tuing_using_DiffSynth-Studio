"""Loss: BCEWithLogitsLoss (multi-label).

论文: binary cross-entropy with logits over the multi-label target.
不要在 loss 前手动 sigmoid. BCEWithLogitsLoss 内部用 log-sum-exp 数值稳定.

  logits: [B, num_classes]   (raw)
  labels: [B, num_classes]   (multi-hot, float 0/1)
  loss   = mean over (B, num_classes)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BCEWithLogitsLoss(nn.Module):
    """封装 BCEWithLogitsLoss, 可选 pos_weight 处理类别不平衡."""

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        # pos_weight shape: [num_classes] (每类的正样本权重)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], labels: [B, C] (float)
        assert logits.shape == labels.shape, (
            f"logits {logits.shape} != labels {labels.shape}"
        )
        return self.loss_fn(logits, labels.float())


def compute_pos_weight(label_tensor: torch.Tensor) -> torch.Tensor:
    """根据数据集统计 pos_weight = neg_count / pos_count (每类)."""
    # label_tensor: [N, C]
    pos = label_tensor.sum(dim=0).clamp(min=1.0)
    neg = (label_tensor.shape[0] - pos).clamp(min=1.0)
    return (neg / pos)
