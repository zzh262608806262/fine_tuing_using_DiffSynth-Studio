"""三方判别器 (cls 多阈值 + qwen3_vl + gpt4o) vs 原标签 (prompt_labels) 对比.

原标签来源:
  - base/lora/quant/distill: outputs/safesora_gen/prompts_sampled_200.json (含 100 unsafe + 100 safe 采样)
  - 各 method 视频文件名 = prompt_id + .mp4; 部分 prompt_id 不在 sampled_200, 需用 prompts_test.json (1471 全集) 兜底
  - malicious: outputs/safesora_gen/prompts_unsafe_181_noCA.json (全部 unsafe)

二分类口径:
  - 原标签 binary_unsafe = any(prompt_labels.values()) == True
  - 判别器 binary_unsafe = pred_unsafe (cls) / pred_unsafe (qwen/gpt4o)
  - 无 pred_unsafe 或 error/parse_failed -> 排除

输出: outputs/safesora_gen/multi_judge/gt_comparison/
  - per_judge_metrics.json: 每方法每判别器的 TP/FP/TN/FN/acc/prec/rec/F1
  - per_method_summary.md: 总览表格
  - all_judges_vs_gt.json: 视频级三方+原标签对比 (含误判样本身份)
  - mislabel_candidates.json: 原标签 unsafe 但所有判别器都判 safe 的样本 (标注可疑)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve()
for _p in [_THIS_DIR.parents[3], _THIS_DIR.parents[2]]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from classify.utils import get_logger

logger = get_logger("compare_gt")


# ---------- ground truth ----------
def load_gt(prompts_files: List[str]) -> Dict[str, dict]:
    """合并多个 prompts_*.json -> {prompt_id: item}."""
    gt: Dict[str, dict] = {}
    for pf in prompts_files:
        if not os.path.exists(pf):
            logger.warning(f"prompts 文件不存在: {pf}")
            continue
        with open(pf, "r", encoding="utf-8") as f:
            d = json.load(f)
        for x in d:
            pid = x.get("prompt_id")
            if pid and pid not in gt:
                gt[pid] = x
    return gt


def is_gt_unsafe(item: dict) -> Optional[bool]:
    labels = item.get("prompt_labels") or {}
    if not isinstance(labels, dict) or not labels:
        return None
    return any(bool(v) for v in labels.values())


def gt_true_labels(item: dict) -> List[str]:
    labels = item.get("prompt_labels") or {}
    return sorted(k for k, v in labels.items() if v)


# ---------- judge results ----------
def load_judge_results(path: str) -> Dict[str, dict]:
    """key = video basename (with .mp4) -> result dict."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx = {}
    for r in data.get("results", []):
        v = r.get("video")
        if v:
            idx[v] = r
    return idx


def judge_pred_unsafe(r: dict) -> Optional[bool]:
    if r.get("error") or r.get("parse_failed"):
        return None
    if "pred_unsafe" in r:
        return bool(r["pred_unsafe"])
    return None


# ---------- metrics ----------
def compute_metrics(gt: List[bool], pred: List[bool]) -> dict:
    """二分类 (unsafe=positive) TP/FP/TN/FN + acc/prec/rec/F1."""
    tp = fp = tn = fn = 0
    for g, p in zip(gt, pred):
        if g and p:
            tp += 1
        elif (not g) and p:
            fp += 1
        elif (not g) and (not p):
            tn += 1
        elif g and (not p):
            fn += 1
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4),
    }


# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/safesora_gen/multi_judge")
    p.add_argument("--methods", default="base,lora,quant,distill,malicious")
    p.add_argument("--judges", default="cls_thr_0.20,cls_thr_0.30,cls_thr_0.50,qwen3_vl,gpt4o",
                   help="判别器名 (对应子目录下文件名不含 .json)")
    p.add_argument("--prompts", default="outputs/safesora_gen/prompts_sampled_200.json,outputs/safesora_gen/prompts_test.json,outputs/safesora_gen/prompts_unsafe_181_noCA.json",
                   help="原标签 prompts json 文件列表 (逗号分隔)")
    p.add_argument("--out_dir", default="outputs/safesora_gen/multi_judge/gt_comparison")
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    judges = [j.strip() for j in args.judges.split(",")]
    prompts_files = [s.strip() for s in args.prompts.split(",")]

    gt = load_gt(prompts_files)
    logger.info(f"加载原标签: {len(gt)} 条 prompts")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_judge_metrics: Dict[str, Dict[str, dict]] = {}
    all_records: Dict[str, List[dict]] = {m: [] for m in methods}
    mislabel_candidates: List[dict] = []

    for method in methods:
        per_judge_metrics[method] = {}
        # 加载此 method 的所有判别器结果
        judge_data: Dict[str, Dict[str, dict]] = {}
        for jn in judges:
            jp = os.path.join(args.root, method, f"{jn}.json")
            judge_data[jn] = load_judge_results(jp)

        # 视频列表 (取并集)
        videos = set()
        for jd in judge_data.values():
            videos.update(jd.keys())
        videos = sorted(videos)

        # 逐视频对照原标签
        for vid in videos:
            pid = vid[:-4] if vid.endswith(".mp4") else vid  # 去 .mp4 -> prompt_id
            gitem = gt.get(pid)
            if gitem is None:
                continue  # 无原标签, 跳过
            g_unsafe = is_gt_unsafe(gitem)
            if g_unsafe is None:
                continue
            rec = {
                "video": vid, "method": method,
                "gt_unsafe": g_unsafe,
                "gt_labels": gt_true_labels(gitem),
            }
            preds = {}
            for jn, jd in judge_data.items():
                r = jd.get(vid)
                if r is None:
                    preds[jn] = None
                else:
                    preds[jn] = judge_pred_unsafe(r)
            rec["judges"] = preds
            all_records[method].append(rec)

            # 候选误标: 原标签 unsafe 但所有判别器都判 safe (排除 None)
            if g_unsafe:
                non_none = [v for v in preds.values() if v is not None]
                if non_none and all(v is False for v in non_none):
                    mislabel_candidates.append(rec)

        # 计算每判别器指标
        for jn in judges:
            g_list, p_list = [], []
            for rec in all_records[method]:
                pv = rec["judges"].get(jn)
                if pv is None:
                    continue
                g_list.append(rec["gt_unsafe"])
                p_list.append(pv)
            per_judge_metrics[method][jn] = compute_metrics(g_list, p_list)

    # 写文件
    with open(out_dir / "per_judge_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_judge_metrics, f, ensure_ascii=False, indent=2)

    with open(out_dir / "all_judges_vs_gt.json", "w", encoding="utf-8") as f:
        json.dump({m: recs for m, recs in all_records.items()}, f, ensure_ascii=False, indent=2)

    with open(out_dir / "mislabel_candidates.json", "w", encoding="utf-8") as f:
        json.dump(mislabel_candidates, f, ensure_ascii=False, indent=2)

    # 生成 markdown 总览
    md_lines = ["# 判别器 vs 原标签 对比报告\n",
                "## 1. 二分类指标 (unsafe=positive)\n",
                "对每个方法每个判别器, 仅取有原标签 + 该判别器有有效预测的视频计算.\n"]
    for method in methods:
        md_lines.append(f"\n### {method}\n")
        md_lines.append("| judge | n | TP | FP | TN | FN | acc | prec | rec | F1 |")
        md_lines.append("|-------|---|----|----|----|----|-----|------|-----|-----|")
        for jn in judges:
            m = per_judge_metrics[method].get(jn, {})
            if not m:
                continue
            md_lines.append(
                f"| {jn} | {m['n']} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} "
                f"| {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
            )

    md_lines.append("\n## 2. 原标签 unsafe 但所有判别器都判 safe (标注可疑)\n")
    md_lines.append(f"共 {len(mislabel_candidates)} 条 (各方法合计). 详见 mislabel_candidates.json\n")
    # 按方法统计
    by_method: Dict[str, int] = {}
    for r in mislabel_candidates:
        by_method[r["method"]] = by_method.get(r["method"], 0) + 1
    md_lines.append("\n| method | mislabel_count |")
    md_lines.append("|--------|-----------------|")
    for m in methods:
        md_lines.append(f"| {m} | {by_method.get(m, 0)} |")

    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"输出: {out_dir}/per_judge_metrics.json, all_judges_vs_gt.json, mislabel_candidates.json, report.md")

    # 控制台简报
    print("\n=== 简报 (F1 / recall / precision) ===")
    for method in methods:
        print(f"\n[{method}]")
        for jn in judges:
            m = per_judge_metrics[method].get(jn, {})
            if m:
                print(f"  {jn:14s} n={m['n']:4d} acc={m['accuracy']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f} F1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
