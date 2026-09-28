"""Safety Classifier 推理: 单视频 CLI + 批量 API (REINS 兼容).

CLI 用法:
  python -m classify.inference.predict \
      --video path/to/video.mp4 \
      --checkpoint outputs/safesora/best.pt

输出 JSON:
  {
    "video": "...",
    "predictions": {"safe": 0.01, "violence": 0.93, ...},
    "predicted_labels": ["violence", "weapon"],
    "unsafe": true
  }

REINS 批量 API (第十四节, 用于 Section 2.3 SPCA):
  predictor = SafetyPredictor(checkpoint)
  R, Y, probs = predictor.predict_for_reins(video_paths)
  # R:     [N, D]  video-level representation (temporal mean pool, D=768)
  # Y:     [N, 2]  [safe, unsafe] 二值 (unsafe = not safe)
  # probs: [N, C]  per-class sigmoid 概率
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve()
for _p in [_THIS_DIR.parents[3], _THIS_DIR.parents[2]]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from classify.factory import build_model, build_transform, build_sampler
from classify.datasets.video_dataset import load_video_frames
from classify.utils import get_logger, set_seed


class SafetyPredictor:
    """加载 checkpoint, 提供单/批量推理 + REINS 兼容 API."""

    def __init__(
        self,
        checkpoint: str,
        device: Optional[str] = None,
        threshold: float = 0.5,
        batch_size: int = 8,
        unsafe_rule: str = "any_class",
    ) -> None:
        """unsafe_rule: unsafe 判定口径。默认 "any_class" 即历史行为 —— 勿改默认值,
        其他项目与 REINS API 依赖此口径。

          - "any_class"      任一 unsafe 类别概率 >= threshold   (历史默认)
          - "one_minus_safe" 1 - p(safe) >= threshold

        Exp 010 实测 (SafeSora test 5745 条): one_minus_safe 的 ROC-AUC 0.941 /
        最优 F1 0.704@thr0.356, 优于 any_class 的 0.931 / 0.666@thr0.739。
        详见 memory/safesora_label_audit.md
        """
        self.logger = get_logger("predict")
        set_seed(42)
        if unsafe_rule not in ("any_class", "one_minus_safe"):
            raise ValueError(f"unsafe_rule 必须是 any_class 或 one_minus_safe, 收到 {unsafe_rule!r}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold
        self.unsafe_rule = unsafe_rule
        self.batch_size = batch_size

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = ckpt["config"]
        self.label_names: List[str] = ckpt["label_names"]
        self.num_classes = len(self.label_names)

        model, _ = build_model(self.cfg)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model = model.to(self.device).eval()
        self.transform = build_transform(model, self.cfg)
        self.sampler = build_sampler(self.cfg)

    # ---------- 单视频 -> frames tensor ----------
    def _load_frames(self, video_path: str) -> Optional[torch.Tensor]:
        frames = load_video_frames(
            video_path, self.sampler,
            self.cfg["video"].get("decode_backend", "av"),
            self.cfg["video"].get("max_decode_attempts", 2),
        )
        if frames is None:
            return None
        return self.transform(frames)  # [T, 3, H, W]

    @torch.no_grad()
    def _forward_batch(self, frames_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """frames_batch: [B, T, 3, H, W] -> (probs [B,C], repr [B,D])"""
        frames_batch = frames_batch.to(self.device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits, video_repr, _ = self.model(frames_batch, return_features=True)
        probs = torch.sigmoid(logits.float())
        return probs.cpu(), video_repr.cpu()

    # ---------- 单视频推理 ----------
    def predict_one(self, video_path: str) -> Optional[Dict]:
        frames = self._load_frames(video_path)
        if frames is None:
            self.logger.warning(f"无法读取视频: {video_path}")
            return None
        frames_batch = frames.unsqueeze(0)  # [1, T, 3, H, W]
        probs, _ = self._forward_batch(frames_batch)
        return self._format_result(video_path, probs[0])

    # ---------- 批量推理 ----------
    @torch.no_grad()
    def predict_batch(self, video_paths: List[str]) -> List[Optional[Dict]]:
        """批量预测, 返回每个视频的标准结果 dict (失败项为 None)."""
        results: List[Optional[Dict]] = [None] * len(video_paths)
        buf: List[Tuple[int, str, torch.Tensor]] = []  # (idx, video_path, frames)

        def flush():
            if not buf:
                return
            batch = torch.stack([b[2] for b in buf], dim=0)  # [B, T, 3, H, W]
            probs, _ = self._forward_batch(batch)
            for j, (idx, vp, _) in enumerate(buf):
                results[idx] = self._format_result(vp, probs[j])
            buf.clear()

        for i, vp in enumerate(video_paths):
            frames = self._load_frames(vp)
            if frames is None:
                results[i] = None
                continue
            buf.append((i, vp, frames))
            if len(buf) >= self.batch_size:
                flush()
        flush()
        return results

    def _decide_unsafe(self, probs_np, pred) -> bool:
        """按 self.unsafe_rule 判定 unsafe。index 0 = safe 类。"""
        if self.num_classes <= 1:
            return bool(pred[0] == 0)
        if self.unsafe_rule == "one_minus_safe":
            return bool((1.0 - float(probs_np[0])) >= self.threshold)
        # any_class: 任一 unsafe 类别命中
        return bool(pred[1:].sum() > 0)

    def _format_result(self, video_path: str, probs: torch.Tensor) -> Dict:
        """构造标准输出 dict. probs: [C] tensor."""
        probs_np = probs.numpy()
        pred = (probs_np >= self.threshold).astype(int)
        predicted_labels = [self.label_names[c] for c in range(self.num_classes) if pred[c] == 1]
        unsafe = self._decide_unsafe(probs_np, pred)
        return {
            "video": video_path,
            "predictions": {self.label_names[c]: float(probs_np[c]) for c in range(self.num_classes)},
            "predicted_labels": predicted_labels,
            "unsafe": unsafe,
        }

    # ---------- REINS 兼容 API ----------
    @torch.no_grad()
    def predict_for_reins(
        self,
        video_paths: List[str],
        return_probs: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """批量处理 Wan2.2 生成视频, 输出 REINS SPCA 所需矩阵.

        Returns:
            R:     [N, D]  video-level representation (temporal mean pool)
            Y:     [N, 2]  [safe, unsafe] 二值 (unsafe = 1 - safe)
            probs: [N, C]  per-class sigmoid 概率 (return_probs=True 时)
        """
        idxs_valid: List[int] = []
        reprs: List[np.ndarray] = []
        probs_all: List[np.ndarray] = []
        buf: List[Tuple[int, torch.Tensor]] = []

        def flush():
            if not buf:
                return
            idxs = [b[0] for b in buf]
            batch = torch.stack([b[1] for b in buf], dim=0).to(self.device)
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits, video_repr, _ = self.model(batch, return_features=True)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            rep = video_repr.cpu().numpy()
            for j, idx in enumerate(idxs):
                idxs_valid.append(idx)
                reprs.append(rep[j])
                probs_all.append(probs[j])

        for i, vp in enumerate(video_paths):
            frames = self._load_frames(vp)
            if frames is None:
                self.logger.warning(f"REINS: 跳过无法读取视频 {vp}")
                continue
            buf.append((i, frames))
            if len(buf) >= self.batch_size:
                flush()
                buf = []
        flush()

        if not reprs:
            D = self.model.backbone.hidden_dim
            empty_R = np.zeros((0, D), dtype=np.float32)
            empty_Y = np.zeros((0, 2), dtype=np.float32)
            return empty_R, empty_Y, (np.zeros((0, self.num_classes), dtype=np.float32) if return_probs else None)

        R = np.stack(reprs, axis=0).astype(np.float32)           # [N, D]
        probs_arr = np.stack(probs_all, axis=0).astype(np.float32)  # [N, C]
        # Y[:,0]=safe, Y[:,1]=unsafe
        pred = (probs_arr >= self.threshold).astype(np.int32)
        if self.unsafe_rule == "one_minus_safe" and self.num_classes > 1:
            unsafe = ((1.0 - probs_arr[:, 0]) >= self.threshold).astype(np.int32)
        else:
            unsafe = (pred[:, 1:].sum(axis=1) > 0).astype(np.int32) if self.num_classes > 1 else (1 - pred[:, 0])
        safe = 1 - unsafe
        Y = np.stack([safe, unsafe], axis=1).astype(np.float32)  # [N, 2]
        return R, Y, (probs_arr if return_probs else None)


# =====================================================================
# CLI
# =====================================================================
def main():
    p = argparse.ArgumentParser(description="Safety Classifier single-video inference")
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--unsafe_rule", type=str, default="any_class",
                   choices=["any_class", "one_minus_safe"],
                   help="unsafe 判定口径; 默认 any_class = 历史行为")
    args = p.parse_args()

    predictor = SafetyPredictor(args.checkpoint, threshold=args.threshold or 0.5,
                                unsafe_rule=args.unsafe_rule)
    result = predictor.predict_one(args.video)
    if result is None:
        print(json.dumps({"video": args.video, "error": "failed to read video"}, ensure_ascii=False))
        sys.exit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
