"""对比两个分类器在同一批视频上的预测结果, 输出一致性报告.

用法:
  python -m classify.evaluation.compare_two_classifiers \
      --result1 outputs/wan5b_porn_predictions/cls1.json \
      --result2 outputs/wan5b_porn_predictions/cls2.json \
      --output_json outputs/wan5b_porn_predictions/compare_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_results(result: list) -> Dict[str, dict]:
    idx = {}
    for r in result:
        key = r.get("video") or r.get("full_path")
        if key:
            idx[key] = r
    return idx


def main():
    p = argparse.ArgumentParser(description="Compare two classifier results")
    p.add_argument("--result1", type=str, required=True, help="第一个分类器的 results.json")
    p.add_argument("--result2", type=str, required=True, help="第二个分类器的 results.json")
    p.add_argument("--output_json", type=str, required=True)
    args = p.parse_args()

    data1 = load(args.result1)
    data2 = load(args.result2)

    meta1 = data1.get("meta", {})
    meta2 = data2.get("meta", {})
    stats1 = data1.get("stats", {})
    stats2 = data2.get("stats", {})

    idx1 = index_results(data1.get("results", []))
    idx2 = index_results(data2.get("results", []))

    common_keys = sorted(set(idx1.keys()) & set(idx2.keys()))
    only1 = sorted(set(idx1.keys()) - set(idx2.keys()))
    only2 = sorted(set(idx2.keys()) - set(idx1.keys()))

    # 二分类一致性
    agree_safe = 0
    agree_unsafe = 0
    disagree_r1s_r2u = 0  # R1 safe, R2 unsafe
    disagree_r1u_r2s = 0  # R1 unsafe, R2 safe
    per_video: List[dict] = []

    for k in common_keys:
        r1 = idx1[k]
        r2 = idx2[k]
        u1 = r1.get("pred_unsafe")
        u2 = r2.get("pred_unsafe")
        agreement = None
        if u1 is not None and u2 is not None:
            if not u1 and not u2:
                agree_safe += 1; agreement = "both_safe"
            elif u1 and u2:
                agree_unsafe += 1; agreement = "both_unsafe"
            elif not u1 and u2:
                disagree_r1s_r2u += 1; agreement = "cls1_safe_cls2_unsafe"
            else:
                disagree_r1u_r2s += 1; agreement = "cls1_unsafe_cls2_safe"

        per_video.append({
            "video": k,
            "cls1_labels": r1.get("pred_labels", []),
            "cls1_unsafe": u1,
            "cls2_labels": r2.get("pred_labels", []),
            "cls2_unsafe": u2,
            "agreement": agreement,
        })

    total_compared = agree_safe + agree_unsafe + disagree_r1s_r2u + disagree_r1u_r2s
    agree_total = agree_safe + agree_unsafe
    disagreement_rate = round(1 - agree_total / max(1, total_compared), 4)

    report = {
        "classifier1": {
            "checkpoint": meta1.get("checkpoint"),
            "name": meta1.get("checkpoint_basename") or Path(args.result1).stem,
            "tag": meta1.get("tag", ""),
            "stats": stats1,
        },
        "classifier2": {
            "checkpoint": meta2.get("checkpoint"),
            "name": meta2.get("checkpoint_basename") or Path(args.result2).stem,
            "tag": meta2.get("tag", ""),
            "stats": stats2,
        },
        "comparison": {
            "common_videos": len(common_keys),
            "only_in_cls1": len(only1),
            "only_in_cls2": len(only2),
            "agree_both_safe": agree_safe,
            "agree_both_unsafe": agree_unsafe,
            "disagree_cls1_safe_cls2_unsafe": disagree_r1s_r2u,
            "disagree_cls1_unsafe_cls2_safe": disagree_r1u_r2s,
            "binary_agreement_rate": round(agree_total / max(1, total_compared), 4),
            "binary_disagreement_rate": disagreement_rate,
            "unsafe_rate_diff": round(
                (stats2.get("unsafe_rate", 0) - stats1.get("unsafe_rate", 0)), 4
            ),
        },
        "per_video": per_video,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    c = report["comparison"]
    print("=== 二分类对比报告 ===")
    print(f"  共同视频数: {c['common_videos']}")
    print(f"  一致性: 都判safe={c['agree_both_safe']}, 都判unsafe={c['agree_both_unsafe']}, 合计agree={c['binary_agreement_rate']:.1%}")
    print(f"  分歧: cls1判safe但cls2判unsafe={c['disagree_cls1_safe_cls2_unsafe']}, 相反={c['disagree_cls1_unsafe_cls2_safe']}")
    print(f"  unsafe率差: cls1={report['classifier1']['stats'].get('unsafe_rate'):.1%} vs cls2={report['classifier2']['stats'].get('unsafe_rate'):.1%} (diff={c['unsafe_rate_diff']:+.1%})")
    print(f"  报告输出: {args.output_json}")


if __name__ == "__main__":
    main()
