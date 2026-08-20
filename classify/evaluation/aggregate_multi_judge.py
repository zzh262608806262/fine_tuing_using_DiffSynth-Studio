"""聚合多判别器输出: per-method summary + cross-judge 一致性 + report.md.

用法:
  python -m classify.evaluation.aggregate_multi_judge \
      --root outputs/safesora_gen/multi_judge \
      --methods base,lora,quant,distill,malicious
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from tabulate import tabulate

_THIS_DIR = Path(__file__).resolve()
for _p in [_THIS_DIR.parents[3], _THIS_DIR.parents[2]]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from classify.utils import get_logger


def load_result(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_results(data: dict) -> Dict[str, dict]:
    """key 是视频 basename."""
    idx = {}
    for r in data.get("results", []):
        key = r.get("video")
        if key:
            idx[key] = r
    return idx


def judge_name_from_path(path: str) -> str:
    return Path(path).stem


def binary_unsafe(r: dict) -> bool | None:
    if r.get("error") or r.get("parse_failed"):
        return None
    # cls 结果: pred_unsafe
    if "pred_unsafe" in r:
        return bool(r["pred_unsafe"])
    return None


def collect_judge_files(root: str, method: str) -> List[str]:
    mdir = os.path.join(root, method)
    if not os.path.isdir(mdir):
        return []
    return sorted(
        os.path.join(mdir, f) for f in os.listdir(mdir)
        if f.endswith(".json") and os.path.isfile(os.path.join(mdir, f))
    )


def build_per_method(root: str, methods: List[str]) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    for method in methods:
        summary[method] = {}
        for jf in collect_judge_files(root, method):
            jname = judge_name_from_path(jf)
            try:
                data = load_result(jf)
            except Exception as e:
                summary[method][jname] = {"error": f"load fail: {e}"}
                continue
            s = dict(data.get("stats", {}))
            m = dict(data.get("meta", {}))
            s["_source_file"] = os.path.relpath(jf, root)
            s["_judge"] = m.get("judge") or "classifier"
            s["_model"] = m.get("model") or m.get("checkpoint_basename") or ""
            s["_threshold"] = m.get("threshold")
            summary[method][jname] = s
    return summary


def build_cross_comparison(root: str, methods: List[str]) -> Dict[str, Dict]:
    comp: Dict[str, Dict] = {}
    for method in methods:
        jfiles = collect_judge_files(root, method)
        if not jfiles:
            continue
        jnames = [judge_name_from_path(p) for p in jfiles]
        datas = {}
        for p, n in zip(jfiles, jnames):
            datas[n] = index_results(load_result(p))

        keys_all = set()
        for idx in datas.values():
            keys_all.update(idx.keys())
        keys_all = sorted(keys_all)

        # 两两 binary agreement
        pairs = {}
        for a, b in itertools.combinations(jnames, 2):
            ia, ib = datas[a], datas[b]
            aa, uu, au, ua, missing = 0, 0, 0, 0, 0
            for k in keys_all:
                ra, rb = ia.get(k), ib.get(k)
                va, vb = binary_unsafe(ra) if ra else None, binary_unsafe(rb) if rb else None
                if va is None or vb is None:
                    missing += 1; continue
                if not va and not vb: aa += 1
                elif va and vb: uu += 1
                elif not va and vb: au += 1
                else: ua += 1
            tot = aa + uu + au + ua
            agree = aa + uu
            pairs[f"{a}__vs__{b}"] = {
                "total_valid": tot,
                "missing_or_failed": missing,
                "agree_safe_safe": aa,
                "agree_unsafe_unsafe": uu,
                "disagree_a_safe_b_unsafe": au,
                "disagree_a_unsafe_b_safe": ua,
                "agreement_rate": round(agree / tot, 4) if tot else 0,
            }

        # 多数投票 + 分歧列表
        per_vid = []
        for k in keys_all:
            votes = {}
            valid_judges = 0
            safe_votes = 0
            unsafe_votes = 0
            for n in jnames:
                r = datas[n].get(k)
                v = binary_unsafe(r) if r else None
                if v is None:
                    votes[n] = None
                    continue
                valid_judges += 1
                votes[n] = "unsafe" if v else "safe"
                if v: unsafe_votes += 1
                else: safe_votes += 1
            if valid_judges == 0:
                majority = None
            elif unsafe_votes > safe_votes:
                majority = "unsafe"
            elif safe_votes > unsafe_votes:
                majority = "safe"
            else:
                majority = "tie"
            per_vid.append({
                "video": k,
                "votes": votes,
                "valid_judges": valid_judges,
                "safe_votes": safe_votes,
                "unsafe_votes": unsafe_votes,
                "majority": majority,
                "unanimous": (safe_votes == valid_judges and valid_judges > 1) or
                             (unsafe_votes == valid_judges and valid_judges > 1),
            })

        major_counter = Counter(p["majority"] for p in per_vid)
        unan_counter = Counter(p["unanimous"] for p in per_vid)
        disputed = [p for p in per_vid if not p["unanimous"] and p["valid_judges"] >= 2]

        comp[method] = {
            "videos": len(keys_all),
            "judges": jnames,
            "pairs": pairs,
            "majority_distribution": dict(major_counter),
            "unanimous_distribution": {
                "unanimous": unan_counter.get(True, 0),
                "not_unanimous": unan_counter.get(False, 0),
            },
            "disputed_count": len(disputed),
            "disputed_sample": disputed[:50],  # 最多看50条
        }
    return comp


def write_report(
    per_method: Dict[str, Dict],
    cross: Dict[str, Dict],
    out_dir: str,
) -> Path:
    lines: List[str] = []
    lines.append("# Multi-Judge 安全评估报告\n")

    # 1. 每方法 × 判别器 unsafe rate 表
    lines.append("## 1. Unsafe Rate 总览\n")
    all_judges: List[str] = []
    for m, js in per_method.items():
        for jn in js:
            if jn not in all_judges:
                all_judges.append(jn)
    table_rows = []
    for method in per_method:
        row = [method]
        for jn in all_judges:
            entry = per_method.get(method, {}).get(jn, {})
            ur = entry.get("unsafe_rate")
            txt = f"{ur:.1%}" if ur is not None else "—"
            if entry.get("_threshold"):
                txt += f" (t={entry['_threshold']})"
            row.append(txt)
        table_rows.append(row)
    lines.append(tabulate(table_rows, headers=["method"] + all_judges, tablefmt="github"))
    lines.append("")

    # 2. 每方法 × 判别器 详细数字
    lines.append("## 2. 各方法详情\n")
    for method in per_method:
        lines.append(f"### {method}\n")
        rows = []
        for jn, s in per_method[method].items():
            if "error" in s:
                rows.append([jn, s["error"]])
                continue
            rows.append([
                jn,
                s.get("_judge", ""),
                s.get("_model", "") or s.get("_threshold", ""),
                s.get("total", ""),
                s.get("ok", s.get("total")),
                s.get("unsafe_count", ""),
                f"{s.get('unsafe_rate', 0):.1%}",
                json.dumps(s.get("per_class_hits", {}), ensure_ascii=False),
            ])
        lines.append(tabulate(
            rows,
            headers=["judge", "type", "model/thr", "total", "ok", "unsafe_cnt", "unsafe_rate", "per_class"],
            tablefmt="github",
        ))
        lines.append("")

    # 3. Cross comparison
    lines.append("## 3. 判别器两两一致性（Binary Safe/Unsafe）\n")
    for method in cross:
        lines.append(f"### {method}\n")
        c = cross[method]
        rows = []
        for pair, d in c["pairs"].items():
            rows.append([
                pair.replace("__vs__", " vs "),
                d["total_valid"],
                d["agree_safe_safe"],
                d["agree_unsafe_unsafe"],
                d["disagree_a_safe_b_unsafe"],
                d["disagree_a_unsafe_b_safe"],
                f"{d['agreement_rate']:.1%}",
            ])
        lines.append(tabulate(
            rows,
            headers=["pair", "valid", "S/S", "U/U", "S/U", "U/S", "agree_rate"],
            tablefmt="github",
        ))
        md = c["majority_distribution"]
        un = c["unanimous_distribution"]
        lines.append(f"\n- 多数投票分布: safe={md.get('safe', 0)}, unsafe={md.get('unsafe', 0)}, tie={md.get('tie', 0)}, none(无有效判)={md.get(None, 0)}")
        lines.append(f"- 全部一致(≥2判): unanimous={un['unanimous']}, not_unanimous={un['not_unanimous']}")
        lines.append(f"- 分歧条目数(≥2判且不一致): {c['disputed_count']}")
        lines.append("")

    # 4. Disputed sample 表
    lines.append("## 4. 分歧样本节选 (不同判别器给出相反结论)\n")
    for method in cross:
        c = cross[method]
        disp = c.get("disputed_sample", [])
        if not disp:
            lines.append(f"### {method}: 无分歧\n")
            continue
        lines.append(f"### {method} ({len(disp)} sample of {c['disputed_count']})\n")
        rows = []
        for p in disp[:30]:
            votes_str = " ".join(f"{k}={'U' if v=='unsafe' else 'S' if v=='safe' else '?'}" for k, v in p["votes"].items())
            rows.append([p["video"], votes_str, p["valid_judges"], p["majority"] or "—"])
        lines.append(tabulate(rows, headers=["video", "votes", "valid", "majority"], tablefmt="github"))
        lines.append("")

    report_path = Path(out_dir) / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="multi_judge root dir")
    p.add_argument("--methods", required=True, help="逗号分隔 method 列表")
    args = p.parse_args()

    logger = get_logger("aggregate_mj")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    per_method = build_per_method(args.root, methods)
    cross = build_cross_comparison(args.root, methods)

    with open(os.path.join(args.root, "per_method_summary.json"), "w", encoding="utf-8") as f:
        json.dump(per_method, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.root, "cross_judge_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(cross, f, ensure_ascii=False, indent=2)
    report_path = write_report(per_method, cross, args.root)
    logger.info(f"per_method_summary.json -> {os.path.join(args.root, 'per_method_summary.json')}")
    logger.info(f"cross_judge_comparison.json -> {os.path.join(args.root, 'cross_judge_comparison.json')}")
    logger.info(f"report.md -> {report_path}")


if __name__ == "__main__":
    main()
