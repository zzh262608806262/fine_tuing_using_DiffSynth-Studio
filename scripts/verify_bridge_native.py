#!/usr/bin/env python3
"""Exp012 J1 附加闸门：桥转换产出的 DiffSynth 基座 DiT 与原生 DiffSynth 5B DiT 逐键、逐值比对。

用途：convert_wan_5b_bridge.py（diffusers→DiffSynth 键映射）的正确性不只靠 2 条视频的余弦门控，
这里在 CPU 上对全部 transformer 键做 1:1 核对：
  - 键集合完全一致（无多无缺）
  - 每键 shape/dtype 一致，数值逐位一致（bf16 下应为完全相等；若桥转换 dtype≠原生则报不一致）

用法：
  python3 verify_bridge_native.py --converted-dir models/Wan-AI/Wan2.2-TI2V-5B \
      --native-dir /home/x_jiage/jiage/DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B \
      [--rel-tol 1e-2] [--outjson outputs/wan5b_bridge_equiv/native_compare.json]

门控：
  1) 结构硬门控：键集合完全一致（无多无缺）、每键 shape/dtype 一致 —— 任一违反退出码 1。
  2) 数值门控：每键 max(|a-b| / max(|a|,|b|)) <= rel_tol（相对容差；排除 dtype 舍入如 bf16，但能抓住结构错配）。
     （原生 DiT 为 F32、桥转换 --dtype auto 亦为 F32 时，逐位一致 rel_diff=0。）
任一键缺失/shape/dtype 不一致或相对偏差超 rel_tol 时退出码 1（--strict 依赖链自动停链）。
仅学术研究用途。
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def _index_shards(shard_dir: Path) -> dict:
    """扫描 dir 下 diffusion_pytorch_model*.safetensors，构建 key -> {path, dtype, shape} 索引（只读头部）。"""
    index = {}
    files = sorted(shard_dir.glob("diffusion_pytorch_model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{shard_dir} 下没有 diffusion_pytorch_model*.safetensors")
    for f in files:
        with safe_open(f, framework="pt") as handle:
            for key in handle.keys():
                sl = handle.get_slice(key)
                if key in index:
                    raise ValueError(f"重复键 {key}（{index[key]['path']} 与 {f}）")
                index[key] = {"path": f, "dtype": sl.get_dtype(), "shape": tuple(sl.get_shape())}
    return index


def _load_tensor(entry: dict, key: str) -> torch.Tensor:
    with safe_open(entry["path"], framework="pt") as handle:
        return handle.get_tensor(key)


def compare(converted_dir: Path, native_dir: Path, rel_tol: float, outjson: Path) -> int:
    c_idx = _index_shards(converted_dir)
    n_idx = _index_shards(native_dir)

    only_c = sorted(set(c_idx) - set(n_idx))
    only_n = sorted(set(n_idx) - set(c_idx))
    common = sorted(set(c_idx) & set(n_idx))
    mismatches = []
    max_rel = 0.0

    for key in common:
        ce, ne = c_idx[key], n_idx[key]
        if ce["shape"] != ne["shape"] or ce["dtype"] != ne["dtype"]:
            mismatches.append({"key": key, "kind": "meta",
                               "converted": f"{ce['dtype']}{ce['shape']}",
                               "native": f"{ne['dtype']}{ne['shape']}"})
            continue
        ct = _load_tensor(ce, key)
        nt = _load_tensor(ne, key)
        a, b = ct.float(), nt.float()
        denom = torch.maximum(a.abs(), b.abs()).clamp_min(1e-8)
        rel = float((a - b).abs().div(denom).max())
        max_rel = max(max_rel, rel)
        if rel > rel_tol:
            mismatches.append({"key": key, "kind": "value",
                               "max_rel_diff": rel, "rel_tol": rel_tol})
        del ct, nt, a, b

    n_value_fail = sum(1 for m in mismatches if m.get("kind") == "value")
    ok = (not only_c) and (not only_n) and (len(mismatches) == 0)
    print(f"[native-compare] converted={len(c_idx)} native={len(n_idx)} common={len(common)} "
          f"only_converted={len(only_c)} only_native={len(only_n)} "
          f"value_fail={n_value_fail} max_rel_diff={max_rel:.3e}")
    if only_c:
        print("  仅 converted 有:", only_c[:10])
    if only_n:
        print("  仅 native 有:", only_n[:10])
    for m in mismatches[:20]:
        print("  mismatch:", m)
    doc = {
        "converted_dir": str(converted_dir), "native_dir": str(native_dir),
        "n_converted": len(c_idx), "n_native": len(n_idx), "n_common": len(common),
        "only_converted": only_c, "only_native": only_n,
        "mismatches": mismatches, "n_value_fail": n_value_fail, "max_rel_diff": max_rel,
        "overall_pass": ok,
    }
    if outjson:
        outjson.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(outjson) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        import os
        os.replace(tmp, outjson)
        print(f"[native-compare] 报告 -> {outjson}")
    print(f"[native-compare] overall={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--converted-dir", required=True, help="桥转换产出的 DiffSynth 基座 DiT 目录")
    ap.add_argument("--native-dir", required=True, help="原生 DiffSynth 5B DiT 目录")
    ap.add_argument("--rel-tol", type=float, default=1e-2,
                    help="数值相对容差（max(|a-b|/max(|a|,|b|)) 上限；F32 同源时=0）")
    ap.add_argument("--outjson", default="outputs/wan5b_bridge_equiv/native_compare.json")
    args = ap.parse_args()
    sys.exit(compare(Path(args.converted_dir), Path(args.native_dir),
                     args.rel_tol, Path(args.outjson)))


if __name__ == "__main__":
    main()