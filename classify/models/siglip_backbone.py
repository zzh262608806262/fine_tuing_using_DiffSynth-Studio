"""SigLIP vision backbone (frozen).

封装为独立模块, 方便替换 checkpoint.

输入:  frames [B, T, 3, H, W]  (float, 已被 processor 归一化)
输出:  frame_features [B, T, D]   (= 论文中的 (T,d) sequence, D=768)

歧义决议 (见 require.md 第七节 & 论文 Appendix C.2):
  论文同时出现 "patch-level features" 与 "(T,d) sequence".
  SigLIP (google/siglip-base-patch16-224) 输出:
    - last_hidden_state : [B*T, num_patches=196, D=768]   (patch-level features; SigLIP 无 CLS token)
    - pooler_output     : [B*T, D=768]                    (对 patch tokens mean pool + LayerNorm)
  sanity check 要求 SigLIP feature = [B, 8, 768], 即每帧 -> 1 个 D 维 token.

  实现:
    feature_source = "patch_mean" (默认):
        取 last_hidden_state, 在 patch 维做 mean -> [B*T, D] -> [B, T, D]
        这既源自 patch-level features, 又得到 (T,d). 等价于 SigLIP pooler 的 mean 部分
        (不含 LayerNorm), 显式可见.
    feature_source = "pooler_output":
        直接用 pooler_output -> [B, T, D]  (含 mean+LayerNorm).
    feature_source = "cls_token":
        取 last_hidden_state[:, 0] -> [B, T, D]  (SigLIP 无 CLS, 仅为兼容其它 backbone 而保留,
        对 SigLIP 等价于取第 0 个 patch token, 不推荐).

冻结要求:
  - requires_grad 全部 False
  - eval mode, training 时也保持 eval
  - forward 用 torch.no_grad()
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers import SiglipVisionModel, AutoImageProcessor
except ImportError as e:  # pragma: no cover
    raise ImportError("transformers is required: pip install transformers") from e


class SigLIPBackbone(nn.Module):
    """冻结的 SigLIP 视觉塔, 输出每帧 D 维特征."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        hidden_dim: int = 768,
        image_size: int = 224,
        feature_source: str = "patch_mean",
        freeze: bool = True,
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        assert feature_source in ("patch_mean", "pooler_output", "cls_token"), (
            f"Unknown feature_source={feature_source}"
        )
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.feature_source = feature_source
        self.torch_dtype = getattr(torch, dtype)

        self.vision_model = SiglipVisionModel.from_pretrained(
            model_name, torch_dtype=self.torch_dtype
        )
        # 校验 hidden dim 与 checkpoint 一致
        vm_cfg = self.vision_model.config
        actual_dim = getattr(vm_cfg, "hidden_size", None)
        if actual_dim is not None and actual_dim != hidden_dim:
            raise ValueError(
                f"backbone.hidden_dim={hidden_dim} 但 SigLIP vision config.hidden_size={actual_dim}, "
                f"请改 config backbone.hidden_dim 与 model_name 匹配."
            )
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        if freeze:
            self._freeze()

    # ---------- 冻结 ----------
    def _freeze(self) -> None:
        for p in self.vision_model.parameters():
            p.requires_grad_(False)
        self.vision_model.eval()

    def train(self, mode: bool = True):
        """无论外层是否 train, backbone 始终保持 eval (论文要求)."""
        super().train(mode)
        self.vision_model.eval()
        return self

    def assert_frozen(self) -> None:
        """sanity check 用: 确认全部 requires_grad=False."""
        unfrozen = [n for n, p in self.vision_model.named_parameters() if p.requires_grad]
        assert len(unfrozen) == 0, f"SigLIP backbone 存在未冻结参数: {unfrozen}"

    # ---------- 前向 ----------
    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """提取每帧 D 维特征.

        Args:
            frames: [B, T, 3, H, W]  已归一化 (与 SiglipImageProcessor 一致), float
                    推荐 H=W=image_size, 否则 processor 之外需保证尺寸.
        Returns:
            frame_features: [B, T, D]   # 论文中的 (T,d) sequence
        """
        # backbone forward 不需要梯度
        b, t, c, h, w = frames.shape
        assert c == 3, f"expected 3 channels, got {c}"
        flat = frames.reshape(b * t, c, h, w)  # [B*T, 3, H, W]
        # SigLIP vision model 接收 pixel_values
        outputs = self.vision_model(pixel_values=flat.to(self.torch_dtype))
        last_hidden_state = outputs.last_hidden_state  # [B*T, num_patches, D]

        if self.feature_source == "patch_mean":
            # patch-level features -> mean over patches -> [B*T, D]
            feat = last_hidden_state.mean(dim=1)
        elif self.feature_source == "pooler_output":
            feat = outputs.pooler_output  # [B*T, D]
        else:  # cls_token (兼容用, SigLIP 无 CLS)
            feat = last_hidden_state[:, 0, :]  # [B*T, D]

        feat = feat.to(frames.dtype)  # 跟随外层 dtype (amp 时为 float16)
        return feat.reshape(b, t, self.hidden_dim)  # [B, T, D]

    # ---------- processor 暴露 ----------
    def get_image_processor(self):
        """供 VideoTransform 使用 (保证归一化统计量与 backbone 一致)."""
        return self.image_processor
