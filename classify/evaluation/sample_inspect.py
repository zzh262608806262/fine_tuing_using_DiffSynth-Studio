"""抽样人眼检查: 从测试集抽 safe/unsafe 各40条, 跑分类器, 输出结果 JSON + 视频软链.

不修改现有代码。复用 SafetyPredictor 做推理。

用法:
  python -m classify.evaluation.sample_inspect \
      --checkpoint outputs/safesora_safety_classifier/best.pt \
      --test_annotation /home/x_jiage/jiage/datasets/SafeSora-Label/test.jsonl \
      --video_root /home/x_jiage/jiage/datasets/SafeSora \
      --n 40 \
      --output_dir outputs/inspect_sample

输出:
  outputs/inspect_sample/
    results.json       # 每条: video, true_labels, pred_probs, pred_labels, unsafe, correct
    videos/safe/       # 40 个 safe 视频软链
    videos/unsafe/     # 40 个 unsafe 视频软链
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_THIS_DIR = Path(__file__).resolve()
for _p in [_THIS_DIR.parents[3], _THIS_DIR.parents[2]]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from classify.inference.predict import SafetyPredictor
from classify.utils import get_logger


def load_and_split(annotation_path: str) -> tuple:
    """加载 test.jsonl, 按 labels[0] 分 safe / unsafe."""
    safe_items, unsafe_items = [], []
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            labels = item.get("labels", [])
            if len(labels) < 13:
                continue
            is_unsafe = any(v == 1 for v in labels[1:])
            if is_unsafe:
                unsafe_items.append(item)
            else:
                safe_items.append(item)
    return safe_items, unsafe_items


def main():
    p = argparse.ArgumentParser(description="Sample safe/unsafe videos for manual inspection")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test_annotation", type=str, required=True)
    p.add_argument("--video_root", type=str, required=True)
    p.add_argument("--n", type=int, default=40, help="每类抽样数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs/inspect_sample")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    logger = get_logger("sample_inspect")
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载并分组
    safe_items, unsafe_items = load_and_split(args.test_annotation)
    logger.info(f"测试集: safe={len(safe_items)}, unsafe={len(unsafe_items)}")

    # 2. 抽样
    n = min(args.n, len(safe_items), len(unsafe_items))
    safe_sample = random.sample(safe_items, n)
    unsafe_sample = random.sample(unsafe_items, n)
    all_sample = [("safe", item) for item in safe_sample] + [("unsafe", item) for item in unsafe_sample]
    logger.info(f"抽样: safe={n}, unsafe={n}, 总计={2*n}")

    # 3. 创建视频软链目录
    for category in ("safe", "unsafe"):
        (output_dir / "videos" / category).mkdir(parents=True, exist_ok=True)

    # 4. 跑分类器
    predictor = SafetyPredictor(args.checkpoint, threshold=args.threshold, batch_size=8)

    results = []
    for group, item in all_sample:
        video_rel = item["video"]
        video_full = os.path.join(args.video_root, video_rel)
        video_name = Path(video_rel).name

        # 软链
        link_path = output_dir / "videos" / group / video_name
        if not link_path.exists():
            try:
                link_path.symlink_to(os.path.abspath(video_full))
            except FileExistsError:
                pass

        # 推理
        result = predictor.predict_one(video_full)
        if result is None:
            logger.warning(f"跳过无法读取: {video_rel}")
            results.append({
                "group": group,
                "video": video_rel,
                "link": str(link_path.relative_to(output_dir)),
                "true_labels": [i for i, v in enumerate(item["labels"]) if v == 1],
                "error": "failed to read video",
            })
            continue

        # 真值
        true_label_indices = [i for i, v in enumerate(item["labels"]) if v == 1]
        true_label_names = [predictor.label_names[i] for i in true_label_indices]

        # 预测
        pred_labels = result["predicted_labels"]
        pred_unsafe = result["unsafe"]

        # 是否正确 (二分类)
        true_unsafe = any(i > 0 for i in true_label_indices)
        correct = (pred_unsafe == true_unsafe)

        results.append({
            "group": group,
            "video": video_rel,
            "link": str(link_path.relative_to(output_dir)),
            "true_labels": true_label_names,
            "true_unsafe": true_unsafe,
            "pred_probs": result["predictions"],
            "pred_labels": pred_labels,
            "pred_unsafe": pred_unsafe,
            "correct": correct,
            "max_unsafe_prob": max(result["predictions"].get(name, 0) for name in predictor.label_names[1:]),
        })

        status = "OK" if correct else "MISS"
        logger.info(f"[{status}] {group:6s} true={true_label_names} pred={pred_labels} "
                    f"unsafe={pred_unsafe} -> {video_name}")

    # 5. 统计
    n_total = len(results)
    n_correct = sum(1 for r in results if r.get("correct"))
    n_safe = sum(1 for r in results if r["group"] == "safe")
    n_unsafe = sum(1 for r in results if r["group"] == "unsafe")
    n_safe_correct = sum(1 for r in results if r["group"] == "safe" and r.get("correct"))
    n_unsafe_correct = sum(1 for r in results if r["group"] == "unsafe" and r.get("correct"))

    summary = {
        "total": n_total,
        "correct": n_correct,
        "accuracy": round(n_correct / max(1, n_total), 4),
        "safe_total": n_safe,
        "safe_correct": n_safe_correct,
        "safe_accuracy": round(n_safe_correct / max(1, n_safe), 4),
        "unsafe_total": n_unsafe,
        "unsafe_correct": n_unsafe_correct,
        "unsafe_accuracy": round(n_unsafe_correct / max(1, n_unsafe), 4),
        "threshold": args.threshold,
        "seed": args.seed,
        "n_per_group": n,
    }

    # 6. 保存
    output_json = output_dir / "results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    logger.info("==== 抽样检查结果 ====")
    logger.info(f"  二分类 accuracy: {summary['accuracy']:.1%} ({n_correct}/{n_total})")
    logger.info(f"  safe accuracy:   {summary['safe_accuracy']:.1%} ({n_safe_correct}/{n_safe})")
    logger.info(f"  unsafe accuracy: {summary['unsafe_accuracy']:.1%} ({n_unsafe_correct}/{n_unsafe})")
    logger.info(f"  结果保存: {output_json}")
    logger.info(f"  视频软链: {output_dir}/videos/{{safe,unsafe}}/")
    logger.info(f"  人眼检查: 打开 results.json, 按 link 字段找视频文件")


if __name__ == "__main__":
    main()
