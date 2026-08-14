"""Safety Classifier (REINS Appendix C.2).

数据流:
  frames [B, T, 3, H, W]
    -> frozen SigLIP -> [B, T, D]            # (T,d) sequence
    -> + sinusoidal temporal PE
    -> Temporal Transformer (4L, 8H) -> [B, T, D]
    -> mean pool over T -> [B, D]
    -> Linear(D, num_classes) -> [B, num_classes]  # logits (multi-label)

推理:  probs = sigmoid(logits)
训练:  loss  = BCEWithLogitsLoss(logits, labels)   # 不要预先 sigmoid

类别名不进模型, num_classes configurable. label mapping 由外部提供.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .siglip_backbone import SigLIPBackbone
from .temporal_transformer import TemporalTransformerEncoder


class SafetyClassifier(nn.Module):
    def __init__(
        self,
        backbone_cfg: Dict,
        temporal_cfg: Dict,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # 1) 冻结的 SigLIP backbone
        self.backbone = SigLIPBackbone(**backbone_cfg)

        D = self.backbone.hidden_dim
        # 2) Temporal Transformer encoder (内含 sinusoidal temporal PE)
        self.temporal = TemporalTransformerEncoder(
            hidden_dim=D,
            num_layers=temporal_cfg.get("num_layers", 4),
            num_heads=temporal_cfg.get("num_heads", 8),
            mlp_ratio=temporal_cfg.get("mlp_ratio", 4),
            activation=temporal_cfg.get("activation", "gelu"),
            dropout=temporal_cfg.get("dropout", 0.1),
            pos_encoding=temporal_cfg.get("pos_encoding", "sinusoidal"),
        )
        # 3) 分类头: Linear(D, num_classes) -> per-category logits
        self.head = nn.Linear(D, num_classes)

    # ---------- 训练态: 始终保持 backbone eval ----------
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.vision_model.eval()  # backbone 永远 eval
        return self

    # ---------- 前向 ----------
    def forward(
        self,
        frames: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """前向.

        Args:
            frames: [B, T, 3, H, W]  已归一化
            return_features: 是否额外返回中间 (video_repr, frame_feats)
        Returns:
            logits: [B, num_classes]   # raw logits, 训练用 BCEWithLogitsLoss
        """
        frame_feats = self.backbone(frames)        # [B, T, D]  no_grad 内部
        t_out = self.temporal(frame_feats)         # [B, T, D]
        video_repr = t_out.mean(dim=1)             # [B, D]     temporal mean pooling
        logits = self.head(video_repr)             # [B, num_classes]
        if return_features:
            return logits, video_repr, frame_feats
        return logits

    @torch.no_grad()
    def predict(
        self,
        frames: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """推理: 返回 (probabilities, binary_predictions).

        probabilities: [B, num_classes]  sigmoid(logits)
        binary_preds:  [B, num_classes]  (probs >= threshold)
        """
        self.eval()
        logits = self.forward(frames)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).to(probs.dtype)
        return probs, preds

    # ---------- 冻结校验 ----------
    def assert_backbone_frozen(self) -> None:
        self.backbone.assert_frozen()

    def trainable_parameters(self):
        """仅返回 temporal + head 的可训练参数 (用于 optimizer / sanity check)."""
        return list(self.temporal.parameters()) + list(self.head.parameters())
