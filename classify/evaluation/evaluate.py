"""独立评估入口: 在 test split 上跑全部指标, 输出 overall + per-class.

用法:
  python -m diffsynth.classify.evaluation.evaluate \
      --checkpoint outputs/safesora/best.pt \
      --test_annotation data/safesora/test.jsonl \
      --video_root /path/to/videos \
      --output_json eval_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve()
for _p in [_THIS_DIR.parents[3], _THIS_DIR.parents[2]]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diffsynth.classify.factory import build_model, build_transform, build_dataset, build_dataloader
from diffsynth.classify.utils import get_logger, set_seed
from diffsynth.classify.evaluation.metrics import compute_multilabel_metrics


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Safety Classifier evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test_annotation", type=str, required=True)
    p.add_argument("--video_root", type=str, default=None)
    p.add_argument("--config", type=str, default=None,
                   help="可选; 不提供则用 checkpoint 内保存的 config")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--per_class", action="store_true", default=True)
    args = p.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("evaluate")

    # 用 checkpoint 内保存的 config + label_names (保证评估与训练一致)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    label_names = ckpt["label_names"]
    threshold = args.threshold if args.threshold is not None else float(cfg["head"].get("threshold", 0.5))

    # 重建模型并加载权重
    model, _ = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device).eval()

    transform = build_transform(model, cfg)
    test_ds = build_dataset(cfg, args.test_annotation, transform, label_names)
    if args.video_root:
        test_ds.video_root = args.video_root
        cfg["data"]["video_root"] = args.video_root
    loader = build_dataloader(test_ds, cfg, train=False, distributed=False)

    all_probs, all_labels, all_videos = [], [], []
    for batch in loader:
        if batch.get("empty"):
            continue
        frames = batch["frames"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = model(frames)
        probs = torch.sigmoid(logits.float())
        all_probs.append(probs.cpu())
        all_labels.append(batch["labels"])
        all_videos.extend(batch["videos"])

    probs = torch.cat(all_probs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    metrics = compute_multilabel_metrics(
        probs, labels, label_names,
        threshold=threshold,
        compute_auroc=cfg["eval"].get("compute_auroc", True),
        compute_auprc=cfg["eval"].get("compute_auprc", True),
    )

    overall = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    logger.info("==== Overall metrics ====")
    for k, v in overall.items():
        logger.info(f"  {k}: {v}")
    if args.per_class:
        logger.info("==== Per-class metrics ====")
        for name, m in metrics["per_class"].items():
            logger.info(f"  {name}: {m}")

    result = {
        "checkpoint": args.checkpoint,
        "threshold": threshold,
        "num_samples": int(labels.shape[0]),
        "overall": overall,
        "per_class": metrics.get("per_class", {}),
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Saved results -> {args.output_json}")


if __name__ == "__main__":
    main()
