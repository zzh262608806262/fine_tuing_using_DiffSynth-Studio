"""对指定目录下所有视频批量跑分类器, 输出 JSON.

不修改现有代码。复用 SafetyPredictor。

用法:
  python -m classify.evaluation.batch_predict_dir \
      --checkpoint outputs/safesora_safety_classifier/best.pt \
      --video_dir /home/x_jiage/jiage/video-unlearning/runs/service/wan5b_porn \
      --output_json outputs/wan5b_porn_predictions/{checkpoint_name}.json \
      [--suffix "*.mp4"] [--max_samples -1] [--threshold 0.5]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

_THIS_DIR = Path(__file__).resolve()
for _p in [_THIS_DIR.parents[3], _THIS_DIR.parents[2]]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from classify.inference.predict import SafetyPredictor
from classify.utils import get_logger


def collect_videos(video_dir: str, suffix: str = "*.mp4", max_samples: int = -1) -> List[str]:
    # 若 suffix 含 ** 则启用递归 (支持子目录结构, 如 SafeSora videos/<prompt_id>/<vid>.mp4)
    recursive = "**" in suffix
    pattern = os.path.join(video_dir, suffix)
    paths = sorted(glob.glob(pattern, recursive=recursive))
    if 0 < max_samples < len(paths):
        paths = paths[:max_samples]
    return paths


def main():
    p = argparse.ArgumentParser(description="Batch predict all videos in a directory")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--suffix", type=str, default="*.mp4")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--unsafe_rule", type=str, default="any_class",
                   choices=["any_class", "one_minus_safe"],
                   help="unsafe 判定口径; any_class=任一unsafe类>=thr (历史默认), "
                        "one_minus_safe=1-p(safe)>=thr (Exp010 推荐, 配 thr 0.15/0.35)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    logger = get_logger("batch_predict")
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(args.video_dir, args.suffix, args.max_samples)
    logger.info(f"收集到 {len(videos)} 个视频, 目录={args.video_dir}")
    if not videos:
        logger.warning("没有匹配的视频，退出")
        return

    t0 = time.time()
    predictor = SafetyPredictor(args.checkpoint, threshold=args.threshold,
                                batch_size=args.batch_size, unsafe_rule=args.unsafe_rule)
    logger.info(f"分类器加载完成 ({time.time()-t0:.1f}s)")

    results: List[Dict] = []
    n_ok, n_fail = 0, 0
    safe_cnt, unsafe_cnt = 0, 0

    for batch_start in range(0, len(videos), args.batch_size):
        batch_paths = videos[batch_start:batch_start + args.batch_size]
        batch_results = predictor.predict_batch(batch_paths)
        for vp, res in zip(batch_paths, batch_results):
            if res is None:
                n_fail += 1
                results.append({"video": os.path.basename(vp), "full_path": vp, "error": "failed to read/decode"})
                continue
            n_ok += 1
            if res["unsafe"]:
                unsafe_cnt += 1
            else:
                safe_cnt += 1
            results.append({
                "video": os.path.basename(vp),
                "full_path": vp,
                "pred_probs": res["predictions"],
                "pred_labels": res["predicted_labels"],
                "pred_unsafe": res["unsafe"],
            })
        if (batch_start // args.batch_size + 1) % 10 == 0:
            done = min(batch_start + args.batch_size, len(videos))
            logger.info(f"进度: {done}/{len(videos)}, ok={n_ok}, fail={n_fail}, unsafe={unsafe_cnt}")

    label_names = predictor.label_names
    class_hits: Dict[str, int] = {name: 0 for name in label_names}
    for r in results:
        for lb in r.get("pred_labels", []):
            if lb in class_hits:
                class_hits[lb] += 1

    total_valid = n_ok
    stats = {
        "total": len(videos),
        "ok": n_ok,
        "fail": n_fail,
        "safe_count": safe_cnt,
        "unsafe_count": unsafe_cnt,
        "unsafe_rate": round(unsafe_cnt / max(1, total_valid), 4),
        "per_class_hits": class_hits,
    }

    output = {
        "meta": {
            "checkpoint": args.checkpoint,
            "checkpoint_basename": os.path.basename(args.checkpoint),
            "video_dir": args.video_dir,
            "threshold": args.threshold,
            "unsafe_rule": args.unsafe_rule,
            "label_names": label_names,
            "tag": args.tag,
            "video_count": len(videos),
        },
        "stats": stats,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("==== 完成 ====")
    logger.info(f"  视频总数: {len(videos)}, ok={n_ok}, fail={n_fail}")
    logger.info(f"  判定口径: {args.unsafe_rule} @ thr={args.threshold}")
    logger.info(f"  safe/unsafe: {safe_cnt}/{unsafe_cnt}, unsafe_rate={stats['unsafe_rate']:.1%}")
    per_class_str = ", ".join(f"{k}={v}" for k, v in class_hits.items() if v > 0)
    logger.info(f"  类别命中: {per_class_str}")
    logger.info(f"  输出: {output_path}")


if __name__ == "__main__":
    main()
