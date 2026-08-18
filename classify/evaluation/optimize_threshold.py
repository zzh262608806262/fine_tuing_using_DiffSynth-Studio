"""阈值优化: 对现有 checkpoint 在测试集上搜索最优二分类阈值。

不修改任何现有代码。复用 factory / dataset / model 构建逻辑。

用法:
  python -m classify.evaluation.optimize_threshold \
      --checkpoint outputs/safesora_safety_classifier/best.pt \
      --test_annotation /home/x_jiage/jiage/datasets/SafeSora-Label/test.jsonl \
      --video_root /home/x_jiage/jiage/datasets/SafeSora \
      --output_json outputs/safesora_safety_classifier/binary_threshold.json

输出:
  - 最优二分类阈值 (按 F1 / accuracy 最大化)
  - 对应的 binary accuracy / precision / recall / F1
  - 阈值-指标曲线 (便于画图)
  - per-class 最优阈值 (附赠)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# 让 `python -m` 与直接运行都能 import
_THIS_DIR = Path(__file__).resolve()
for _p in [_THIS_DIR.parents[3], _THIS_DIR.parents[2]]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from classify.factory import build_model, build_transform, build_dataset, build_dataloader
from classify.utils import get_logger, set_seed


@torch.no_grad()
def collect_probs(
    checkpoint: str,
    test_annotation: str,
    video_root: str | None,
    device: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """加载 checkpoint, 在测试集上跑全部样本, 返回 (probs, labels, label_names)."""
    set_seed(42)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger = get_logger("optimize_threshold")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    label_names: List[str] = ckpt["label_names"]

    model, _ = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(dev).eval()

    transform = build_transform(model, cfg)
    test_ds = build_dataset(cfg, test_annotation, transform, label_names)
    if video_root:
        test_ds.video_root = video_root
    loader = build_dataloader(test_ds, cfg, train=False, distributed=False)

    all_probs, all_labels = [], []
    n_total, n_empty = 0, 0
    for batch in loader:
        n_total += 1
        if batch.get("empty"):
            n_empty += 1
            continue
        frames = batch["frames"].to(dev, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = model(frames)
        probs = torch.sigmoid(logits.float()).cpu()
        all_probs.append(probs)
        all_labels.append(batch["labels"])

    logger.info(f"batches: {n_total}, empty: {n_empty}, valid: {n_total - n_empty}")
    probs = torch.cat(all_probs, dim=0).numpy()        # [N, C]
    labels = torch.cat(all_labels, dim=0).numpy()       # [N, C]
    logger.info(f"samples: {probs.shape[0]}, classes: {probs.shape[1]}")
    return probs, labels, label_names


def optimize_binary_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    step: float = 0.01,
) -> Dict:
    """对 binary safe/unsafe 决策做阈值扫描.

    决策逻辑: unsafe_pred = max(probs[:, 1:]) >= threshold
              unsafe_label = sum(labels[:, 1:]) > 0

    Returns:
        dict with keys: best_threshold_by_f1, best_threshold_by_acc,
                        curve (list of {threshold, accuracy, precision, recall, f1}),
                        metrics_at_0_5, metrics_at_best
    """
    C = probs.shape[1]
    assert C > 1, "need at least 2 classes for binary safe/unsafe"

    # 二分类 ground truth: 是否 unsafe
    unsafe_label = (labels[:, 1:].sum(axis=1) > 0).astype(np.int64)  # [N]
    # 对每个样本, 12 个 unsafe 类的最大概率
    max_unsafe_prob = probs[:, 1:].max(axis=1)  # [N]

    thresholds = np.arange(step, 1.0, step)
    curve = []
    best_f1 = -1.0
    best_th_f1 = 0.5
    best_acc = -1.0
    best_th_acc = 0.5

    for t in thresholds:
        unsafe_pred = (max_unsafe_prob >= t).astype(np.int64)
        tp = int(((unsafe_pred == 1) & (unsafe_label == 1)).sum())
        fp = int(((unsafe_pred == 1) & (unsafe_label == 0)).sum())
        fn = int(((unsafe_pred == 0) & (unsafe_label == 1)).sum())
        tn = int(((unsafe_pred == 0) & (unsafe_label == 0)).sum())

        n = tp + fp + fn + tn
        acc = (tp + tn) / max(1, n)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)

        curve.append({
            "threshold": round(float(t), 4),
            "accuracy": round(float(acc), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_th_f1 = float(t)
        if acc > best_acc:
            best_acc = acc
            best_th_acc = float(t)

    def metrics_at(t: float) -> Dict:
        unsafe_pred = (max_unsafe_prob >= t).astype(np.int64)
        tp = int(((unsafe_pred == 1) & (unsafe_label == 1)).sum())
        fp = int(((unsafe_pred == 1) & (unsafe_label == 0)).sum())
        fn = int(((unsafe_pred == 0) & (unsafe_label == 1)).sum())
        tn = int(((unsafe_pred == 0) & (unsafe_label == 0)).sum())
        n = tp + fp + fn + tn
        acc = (tp + tn) / max(1, n)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        return {
            "threshold": round(float(t), 4),
            "accuracy": round(float(acc), 6),
            "precision": round(float(prec), 6),
            "recall": round(float(rec), 6),
            "f1": round(float(f1), 6),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    return {
        "best_threshold_by_f1": round(best_th_f1, 4),
        "best_threshold_by_accuracy": round(best_th_acc, 4),
        "metrics_at_0.5": metrics_at(0.5),
        "metrics_at_best_f1": metrics_at(best_th_f1),
        "metrics_at_best_acc": metrics_at(best_th_acc),
        "curve": curve,
        "n_samples": int(len(unsafe_label)),
        "n_unsafe": int(unsafe_label.sum()),
        "n_safe": int((unsafe_label == 0).sum()),
    }


def optimize_per_class_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    step: float = 0.01,
) -> Dict:
    """附赠: 对每个类单独搜索最优阈值 (最大化 F1)."""
    C = probs.shape[1]
    per_class = {}
    for c in range(C):
        name = label_names[c] if c < len(label_names) else str(c)
        support = int(labels[:, c].sum())
        if support == 0 or support == len(labels):
            per_class[name] = {"best_threshold": None, "support": support, "note": "skipped (no positive or all positive)"}
            continue
        best_f1 = -1.0
        best_t = 0.5
        for t in np.arange(step, 1.0, step):
            pred = (probs[:, c] >= t).astype(np.int64)
            tp = int(((pred == 1) & (labels[:, c] == 1)).sum())
            fp = int(((pred == 1) & (labels[:, c] == 0)).sum())
            fn = int(((pred == 0) & (labels[:, c] == 1)).sum())
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-12, prec + rec)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        # 在最优阈值下的指标
        pred = (probs[:, c] >= best_t).astype(np.int64)
        tp = int(((pred == 1) & (labels[:, c] == 1)).sum())
        fp = int(((pred == 1) & (labels[:, c] == 0)).sum())
        fn = int(((pred == 0) & (labels[:, c] == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        per_class[name] = {
            "best_threshold": round(best_t, 4),
            "best_f1": round(float(2 * prec * rec / max(1e-12, prec + rec)), 6),
            "precision": round(float(prec), 6),
            "recall": round(float(rec), 6),
            "support": support,
        }
    return per_class


def main():
    p = argparse.ArgumentParser(description="Binary safe/unsafe threshold optimization")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test_annotation", type=str, required=True)
    p.add_argument("--video_root", type=str, default=None)
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--step", type=float, default=0.01)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    logger = get_logger("optimize_threshold")

    # 1. 收集所有样本的 probs
    probs, labels, label_names = collect_probs(
        args.checkpoint, args.test_annotation, args.video_root, args.device,
    )

    # 2. 二分类阈值优化
    binary_result = optimize_binary_threshold(probs, labels, step=args.step)

    logger.info("==== Binary safe/unsafe threshold optimization ====")
    logger.info(f"  samples: {binary_result['n_samples']} "
                f"(safe={binary_result['n_safe']}, unsafe={binary_result['n_unsafe']})")
    logger.info(f"  --- threshold=0.5 (current) ---")
    m05 = binary_result["metrics_at_0.5"]
    logger.info(f"    accuracy={m05['accuracy']:.4f}  precision={m05['precision']:.4f}  "
                f"recall={m05['recall']:.4f}  f1={m05['f1']:.4f}  "
                f"(tp={m05['tp']}, fp={m05['fp']}, fn={m05['fn']}, tn={m05['tn']})")
    logger.info(f"  --- best by F1 (threshold={binary_result['best_threshold_by_f1']}) ---")
    mbf = binary_result["metrics_at_best_f1"]
    logger.info(f"    accuracy={mbf['accuracy']:.4f}  precision={mbf['precision']:.4f}  "
                f"recall={mbf['recall']:.4f}  f1={mbf['f1']:.4f}  "
                f"(tp={mbf['tp']}, fp={mbf['fp']}, fn={mbf['fn']}, tn={mbf['tn']})")
    logger.info(f"  --- best by accuracy (threshold={binary_result['best_threshold_by_accuracy']}) ---")
    mba = binary_result["metrics_at_best_acc"]
    logger.info(f"    accuracy={mba['accuracy']:.4f}  precision={mba['precision']:.4f}  "
                f"recall={mba['recall']:.4f}  f1={mba['f1']:.4f}  "
                f"(tp={mba['tp']}, fp={mba['fp']}, fn={mba['fn']}, tn={mba['tn']})")

    # 3. per-class 阈值优化 (附赠)
    per_class = optimize_per_class_threshold(probs, labels, label_names, step=args.step)
    logger.info("==== Per-class best thresholds ====")
    for name, m in per_class.items():
        if m.get("best_threshold") is not None:
            logger.info(f"  {name:25s}: threshold={m['best_threshold']:.4f}  "
                        f"f1={m['best_f1']:.4f}  support={m['support']}")
        else:
            logger.info(f"  {name:25s}: skipped (support={m['support']})")

    # 4. 保存结果
    result = {
        "checkpoint": args.checkpoint,
        "binary": binary_result,
        "per_class": per_class,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Saved -> {args.output_json}")


if __name__ == "__main__":
    main()
