#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_unlearn_lora_wan5b.py
===========================

把 VU（video-unlearning）GradAscent 训练产出的"擦除 LoRA"合并进 diffusers
Wan2.2-TI2V-5B transformer 全量权重，得到"擦除后全量权重"（diffusers transformer
目录，保持可被 diffusers/task1 桥加载），再用 scripts/convert_wan_5b_bridge.py
（Task 1 桥）转成 DiffSynth 原格式的"擦除后基座"（models/Wan-AI/Wan2.2-TI2V-5B-erased）。

数据流
------
    adapter_final.pt (VU GradAscent, video-transformer-lora-v1)
        +  diffusers 5B transformer 基座（diffusion_pytorch_model*.safetensors）
                │  mergre: W_new = W + (alpha/rank) * (up.weight @ down.weight)
                ▼
        merged diffusers transformer（models/unlearn/wan5b_nudity_grad_ascent/merged/）
                │  convert_wan_5b_bridge.py（Task 1 桥，逐键映射 + 哈希复核）
                ▼
        DiffSynth 擦除后基座（models/Wan-AI/Wan2.2-TI2V-5B-erased/）
        （T5/VAE 不属于本脚本职责：复用 models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/，
          由生成侧脚本负责加载。）

--------------------------------------------------------------------------
【VU 适配器结构核对结论】（依据 video-unlearning/src/unlearning/artifacts/
video_transformer_lora.py 逐行核实，及 training/diffusion_base.py 的 save_artifact 分支）
--------------------------------------------------------------------------
1. 文件格式：`torch.save` 的一个 dict，顶层键固定为
       {"format": "video-transformer-lora-v1",
        "config":  asdict(VideoTransformerLoRAConfig),
        "state_dict": {...}}
2. VideoTransformerLoRAConfig（wan 家族默认值）：
       model_family="wan", rank=8, alpha=16.0, layer_start=0, layer_end=29,
       targets=("attn1.to_q","attn1.to_k","attn1.to_v","attn1.to_out.0",
                "attn2.to_q","attn2.to_k","attn2.to_v","attn2.to_out.0"),
       experts=("transformer",)
   （注意：实际参数以 .pt 内嵌 config 为准，本脚本运行时从 payload 读取，不写死。）
3. state_dict 键名约定（关键核对结论）：
       f"{expert}.blocks.<layer>.<target>.down.weight"
       f"{expert}.blocks.<layer>.<target>.up.weight"
   - 前缀即 expert 名 "transformer."（diffusion_base 里 Wan 用
     model.lora_target_denoisers() 取 experts，单 DiT 时 = ("transformer",)）。
   - 无 "pipe.dit." 前缀、无 "lora." 前缀、无嵌套 dict（不存在 {"lora_up": {...}} 包装，
     down/up 是平铺在 state_dict 里的两个键）。
4. LoRALinear 数学语义（video_transformer_lora.py 复用 qwen_text_lora.py 的 LoRALinear）：
       down = nn.Linear(in_features, rank, bias=False)   # down.weight: (rank, in)
       up   = nn.Linear(rank, out_features, bias=False)  # up.weight:   (out, rank)
       scaling = alpha / rank                            # 16/8 = 2.0（默认）
       forward: y = base(x) + scaling * up(down(x))      # 零初始化 up ⇒ 未训练即恒等
   理论合并公式（把 LoRA 残差折叠进 W）：
       W_new = W + scaling * (up.weight @ down.weight)   # (out,in) + (out,rank)x(rank,in)
5. 键映射（adapter → base diffusers）：
       adapter "transformer.blocks.N.<target>.down.weight"
           ⇒ 剥 expert 前缀 "transformer." ⇒ "blocks.N.<target>.down.weight"
           ⇒ 去掉 ".down.weight" ⇒ base 键 "blocks.N.<target>.weight"
   diffusers WanTransformer3DModel 的 state_dict 顶层即 blocks.N.attn1.to_q.weight 等
   （与 task1 桥 scripts/convert_wan_5b_bridge.py 的 gen_diffusers_keys 一致）。
   为兼容带 "transformer."/"model."/"module." 前缀导出的基座，映射时对 base 键做
   带/不带前缀双匹配（见 find_base_key）。

--------------------------------------------------------------------------
用法
--------------------------------------------------------------------------
::

    # 1) dry-run：本地即可跑，打印合并计划 + 桥转换计划，不读 tensor、不写文件
    python3 scripts/merge_unlearn_lora_wan5b.py \\
        --adapter models/unlearn/wan5b_nudity_grad_ascent/adapter_final.pt \\
        --base-dir <diffusers 5B>/transformer \\
        --output-dir models/unlearn/wan5b_nudity_grad_ascent/merged \\
        --bridge-output-dir models/Wan-AI/Wan2.2-TI2V-5B-erased --dry-run

    # 2) 正式（集群）：合并 + 桥转换一步完成
    python3 scripts/merge_unlearn_lora_wan5b.py \\
        --adapter models/unlearn/wan5b_nudity_grad_ascent/adapter_final.pt \\
        --base-dir /proj/berzelius-aiics-real/users/x_jiage/huggingface_cache/hub/\\
            models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/\\
            b8fff7315c768468a5333511427288870b2e9635/transformer \\
        --output-dir models/unlearn/wan5b_nudity_grad_ascent/merged \\
        --bridge-output-dir models/Wan-AI/Wan2.2-TI2V-5B-erased \\
        --bridge-dtype bf16 --strict --check-hash

产物
----
- 合并后 diffusers transformer 目录（config.json / index 等原样复制，只有被合并的
  .safetensors 重写，分片文件名与键集合不变 ⇒ diffusers 可直接加载原 config 与 index）。
- merge_report.json：覆盖键数/形状/dtype/未匹配键/分片重写清单。
- （可选）桥转换产物：DiffSynth 分片 + model_index.json + conversion_report.json。

注意
----
- 形状不匹配 ⇒ 硬失败；dtype 不一致 ⇒ 记录并在 fp32 下计算后回写基座 dtype（bf16 安全）。
- --strict 遇上未匹配的 adapter 键即报错（默认仅警告并写进报告）。
- VU 目录只读（本脚本不 import VU 代码，仅按其文件格式解析 .pt，规避 VU 的
  diffusers 依赖）；不修改 FT specs/ 下任何文档；不写 .md。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open

_KNOWN_EXPERT_PREFIXES = ("transformer.", "model.", "module.")

# VU artifact 格式名（video_transformer_lora.py 的 ARTIFACT_FORMAT）
_ARTIFACT_FORMAT = "video-transformer-lora-v1"

# wan 家族 LoRA 目标投影（仅用于 dry-run 的期望计数核对；实际以 config["targets"] 为准）
_WAN_TARGETS = (
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
)


# ---------------------------------------------------------------------------
# 1. 适配器加载与解析
# ---------------------------------------------------------------------------

def load_adapter(path: str) -> dict:
    """torch.load 适配器（weights_only=True，CPU）；校验顶层结构与 format。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"adapter 不存在: {path}")
    payload = torch.load(p, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"adapter {path} 不是 dict（torch.save 的 payload）")
    fmt = payload.get("format")
    if fmt is None:
        print(f"[警告] adapter {path} 无 'format' 字段，按 video-transformer-lora-v1 结构解析")
    elif fmt != _ARTIFACT_FORMAT:
        raise ValueError(
            f"adapter format={fmt!r}，本脚本只支持 {_ARTIFACT_FORMAT}（GradAscent wan LoRA）"
        )
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"adapter {path} 缺少非空 state_dict")
    bad = [k for k in state if not k.endswith((".down.weight", ".up.weight"))]
    if bad:
        raise ValueError(f"adapter 含非 LoRA 键（期望 ...down.weight / ...up.weight）: {bad[:5]}")
    return payload


def _norm_mapping(module_path: str) -> str:
    """把 adapter 的模块路径（含 expert 前缀）归一化为 diffusers 侧模块路径。

    例: "transformer.blocks.0.attn1.to_q" → "blocks.0.attn1.to_q"
    返回 None 表示无法归一化（非 blocks.* 目标，将计入未匹配）。
    """
    stripped = module_path
    for prefix in _KNOWN_EXPERT_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    if not stripped.startswith("blocks."):
        return None
    return stripped


def parse_adapter_payload(payload: dict):
    """解析 state_dict → {(module_path): {"down": t, "up": t}} 与 scaling。

    返回 (config, experts, scaling, pairs, unmatched_keys)：
    - pairs: {module_path(str): {"down": Tensor(rank,in), "up": Tensor(out,rank)}}
    - unmatched_keys: adapter 键里无法归一到 blocks.* 目标的（如 text encoder 残留）
    """
    config = dict(payload.get("config") or {})
    experts = tuple(config.get("experts") or ("transformer",))
    rank = int(config.get("rank", 8))
    alpha = float(config.get("alpha", 16.0))
    scaling = alpha / rank

    pairs: dict = {}
    unmatched_keys: list = []
    for key, tensor in payload["state_dict"].items():
        # key: "<expert>.blocks.N.<target>.down.weight"；rsplit('.', 2) 拆出 (模块路径, down/up, weight)
        module_path, part, leaf = key.rsplit(".", 2)
        if leaf != "weight" or part not in ("down", "up"):
            unmatched_keys.append(key)
            continue
        norm = _norm_mapping(module_path)
        if norm is None:
            unmatched_keys.append(key)
            continue
        entry = pairs.setdefault(norm, {})
        if part in entry:
            raise ValueError(f"adapter 键重复: {key}")
        entry[part] = tensor

    bad = [m for m, e in pairs.items() if set(e) != {"down", "up"}]
    if bad:
        raise ValueError(f"以下模块缺 down/up 配对: {bad}")

    # 期望配对个数的形式核对（不写死 30 层，按 config 的 layer 区间与 targets 计算）
    expected = None
    if config.get("targets"):
        targets = tuple(config["targets"])
        n_layers = int(config.get("layer_end", 0)) - int(config.get("layer_start", 0)) + 1
        expected = len(targets) * n_layers
    return config, experts, scaling, pairs, unmatched_keys, expected


# ---------------------------------------------------------------------------
# 2. base 键匹配 / 合并
# ---------------------------------------------------------------------------

def find_base_key(base_keys: set, module_path: str) -> str | None:
    """adapter 模块路径 → 基座 state_dict 键（带/不带常见导出前缀双匹配）。"""
    candidates = [module_path + ".weight"]
    for prefix in _KNOWN_EXPERT_PREFIXES:
        candidates.append(prefix + module_path + ".weight")
        if module_path.startswith(prefix):
            candidates.append(module_path[len(prefix):] + ".weight")
    for c in candidates:
        if c in base_keys:
            return c
    return None


def _compute_delta(up, down, scaling):
    """delta = scaling * (up @ down)，统一在 fp32 下计算（bf16/fp16 基座防精度损失）。"""
    up32 = up.to(torch.float32)
    down32 = down.to(torch.float32)
    return (up32 @ down32) * scaling


def _merge_one(base_t, up, down, scaling, module_path):
    """W_new = W + scaling * (up.weight @ down.weight)。先做形状/秩校验。"""
    out_r, rank = tuple(up.shape)
    r2, in_f = tuple(down.shape)
    if rank != r2:
        raise ValueError(f"[{module_path}] rank 不一致: up.cols={rank} vs down.rows={r2}")
    if tuple(base_t.shape) != (out_r, in_f):
        raise ValueError(
            f"[{module_path}] 形状不匹配: base={tuple(base_t.shape)} "
            f"vs up@down={(out_r, in_f)}"
        )
    delta = _compute_delta(up, down, scaling)
    return (base_t.to(torch.float32) + delta).to(base_t.dtype)  # 回写基座 dtype


def load_base_state(files: list) -> "OrderedDict[str, dict]":
    """加载基座全部分片到 CPU。返回 {key: tensor}。"""
    state = OrderedDict()
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as st:
            for k in st.keys():
                state[k] = st.get_tensor(k)
    return state


def merge_pairs(state: dict, pairs: dict, scaling: float):
    """逐键合并; 返回 (merged_records, unmatched) 供报告。

    merged_records: [{base_key, adapter_module, shape, dtype_adapter, dtype_base}]
    unmatched: [{adapter_key 或 module_path, reason}]
    """
    base_keys = set(state)
    merged_records = []
    unmatched = []

    for module_path, entry in pairs.items():
        base_key = find_base_key(base_keys, module_path)
        if base_key is None:
            unmatched.append(
                {"module": module_path, "reason": "基座 state_dict 无对应键（含常见导出前缀）"}
            )
            continue
        base_t = state[base_key]
        merged_records.append({
            "base_key": base_key,
            "adapter_module": module_path,
            "shape": list(base_t.shape),
            "dtype_base": str(base_t.dtype),
            "dtype_adapter": str(entry["up"].dtype),
            "rank": int(entry["up"].shape[1]),
            "scaling": scaling,
        })
        state[base_key] = _merge_one(base_t, entry["up"], entry["down"], scaling, module_path)
    return merged_records, unmatched


# ---------------------------------------------------------------------------
# 3. 输出：复制非张量文件 + 只重写"被合并"分片
# ---------------------------------------------------------------------------

def copy_base_aux_files(base_dir: str, output_dir: str) -> list:
    """把基座目录里除 diffusion_pytorch_model*.safetensors 外的文件/目录原样复制。"""
    copied = []
    for name in sorted(os.listdir(base_dir)):
        src = os.path.join(base_dir, name)
        dst = os.path.join(output_dir, name)
        if name.startswith("diffusion_pytorch_model") and name.endswith(".safetensors"):
            continue
        if os.path.isdir(src):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        copied.append(name)
    return copied


def write_merged_output(base_dir, output_dir, files, state, merged_base_keys):
    """写合并目录：未被合并的分片字节复制，被合并的分片用相同键集合重写。

    返回 (rewrites, copies)：[{file, n_keys, rewritten}]
    """
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)
    aux = copy_base_aux_files(base_dir, output_dir)
    print(f"[复制] 原样复制非张量文件/目录 {len(aux)} 个: {aux}")

    rewrites, copies = [], []
    for f in files:
        fname = os.path.basename(f)
        with safe_open(f, framework="pt", device="cpu") as st:
            shard_keys = list(st.keys())
        touched = [k for k in shard_keys if k in merged_base_keys]
        dst = os.path.join(output_dir, fname)
        if touched:
            tensors = OrderedDict((k, state[k]) for k in shard_keys)
            save_file(tensors, dst, metadata={"format": "pt"})
            rewrites.append({"file": fname, "n_keys": len(shard_keys),
                             "merged_keys": len(touched), "size_gb": round(
                                 os.path.getsize(dst) / (1024 ** 3), 3)})
            print(f"[重写] {fname}: {len(shard_keys)} 键（含 {len(touched)} 个被合并）")
        else:
            shutil.copy2(f, dst)
            copies.append({"file": fname, "n_keys": len(shard_keys), "size_gb": round(
                os.path.getsize(dst) / (1024 ** 3), 3)})
            print(f"[复制] {fname}: 无合并键，字节原样复制")
    return rewrites, copies


# ---------------------------------------------------------------------------
# 4. 桥转换（Task 1, scripts/convert_wan_5b_bridge.py）
# ---------------------------------------------------------------------------

def _import_bridge():
    """import 同目录的 convert_wan_5b_bridge.py（顶层只依赖 stdlib，torch 均为懒加载）。"""
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import convert_wan_5b_bridge  # noqa: PLC0415
    return convert_wan_5b_bridge


def print_bridge_command(merged_dir, bridge_out, bridge_dtype, max_shard, strict, check_hash):
    print("[桥转换计划] 等价命令：")
    out_flag = f"--output-dir {bridge_out}" if bridge_out else "--output-dir <DiffSynth 擦除后基座目录>"
    cmd = (
        f"python3 scripts/convert_wan_5b_bridge.py "
        f"--input-dir {merged_dir} "
        f"{out_flag} "
        f"--dtype {bridge_dtype} --max-shard-size-gb {max_shard}"
    )
    if strict:
        cmd += " --strict"
    if check_hash:
        cmd += " --check-hash"
    print(f"  {cmd}")


def run_bridge(merged_dir, bridge_out, bridge_dtype, max_shard, strict, check_hash):
    import argparse as _argparse

    bridge = _import_bridge()
    args = _argparse.Namespace(
        input_dir=merged_dir,
        output_dir=bridge_out,
        dtype=bridge_dtype,
        max_shard_size_gb=max_shard,
        strict=strict,
        check_hash=check_hash,
    )
    print(f"[桥转换] {merged_dir} → {bridge_out}（dtype={bridge_dtype}）")
    return bridge.run_convert(args)


# ---------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------

def find_base_files(base_dir: str) -> list:
    files = sorted(glob.glob(os.path.join(base_dir, "diffusion_pytorch_model*.safetensors")))
    if not files:
        raise FileNotFoundError(f"基座目录 {base_dir} 无 diffusion_pytorch_model*.safetensors")
    return files


def run_dry_run(args):
    print("=" * 78)
    print("合并 dry-run：擦除 LoRA → diffusers 5B transformer → DiffSynth 基座")
    print("=" * 78)

    print(f"\n[adapter] {args.adapter}")
    if not os.path.isfile(args.adapter):
        print("  ✗ 文件不存在（预期在集群；仍打印后续计划）")
        return 0
    payload = load_adapter(args.adapter)
    config, experts, scaling, pairs, unmatched_keys, expected = parse_adapter_payload(payload)
    print(f"  format       = {payload.get('format')}")
    print(f"  config       = {json.dumps(config, ensure_ascii=False)}")
    print(f"  scaling      = alpha/rank = {scaling}（alpha={config.get('alpha')}, "
          f"rank={config.get('rank')}）")
    print(f"  experts      = {experts}")
    print(f"  配对模块数   = {len(pairs)}（期望 {expected} = targets x 层数）")
    print(f"  adapter 键数 = {len(payload['state_dict'])}")

    print(f"\n[base-dir] {args.base_dir}")
    if not os.path.isdir(args.base_dir):
        print("  ✗ 目录不存在（预期在集群；跳过 base 键对齐核对）")
    else:
        try:
            files = find_base_files(args.base_dir)
            bridge = _import_bridge()
            base_keys_shapes = bridge.read_keys_shapes(files)
            base_keys = set(base_keys_shapes)
            print(f"  基座键数 = {len(base_keys)}（{len(files)} 个分片）")
            matched, unmatched = [], []
            for module_path in sorted(pairs):
                bk = find_base_key(base_keys, module_path)
                (matched if bk else unmatched).append((module_path, bk))
            print(f"  对齐: 匹配 {len(matched)} / 未匹配 {len(unmatched)}")
            print("  [示例 3 条]")
            for mp, bk in matched[:3]:
                print(f"    {mp}.down.weight → {bk}（基座形状 {base_keys_shapes[bk]}）")
            for mp, bk in unmatched[:5]:
                print(f"    ✗ {mp} → 无对应基座键")
            print(f"\n  期望合并公式: W_new = W + {scaling} * (up.weight @ down.weight)")
            print(f"  未匹配 adapter 键: {len(unmatched_keys)} "
                  f"({unmatched_keys[:3] if unmatched_keys else '无'})")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ 读取基座头部失败: {e}")

    print(f"\n[输出计划]")
    print(f"  merged diffusers transformer → {args.output_dir}")
    print(f"    - config.json / index 等原样复制")
    print(f"    - 只有被合并的 diffusion_pytorch_model*.safetensors 重写")
    print(f"    - merge_report.json（覆盖键数/形状/未匹配键）")

    if args.bridge_output_dir:
        print_bridge_command(args.output_dir, args.bridge_output_dir,
                             args.bridge_dtype, args.max_shard_size_gb,
                             args.strict, args.check_hash)
    print("\n[dry-run] 未写任何文件。")
    return 0


def run_merge(args):
    print("=" * 78)
    print("合并：擦除 LoRA → diffusers 5B transformer")
    print("=" * 78)

    # 1) adapter
    payload = load_adapter(args.adapter)
    config, experts, scaling, pairs, unmatched_keys, expected = parse_adapter_payload(payload)
    print(f"[adapter] {args.adapter}")
    print(f"  format={payload.get('format')} config={json.dumps(config, ensure_ascii=False)}")
    print(f"  scaling=alpha/rank={scaling}  配对模块数={len(pairs)}"
          f"（期望 {expected}）  adapter 键数={len(payload['state_dict'])}")

    # 2) base
    files = find_base_files(args.base_dir)
    print(f"[基座] {args.base_dir}（{len(files)} 个分片）")
    state = load_base_state(files)
    print(f"  载入 {len(state)} 个键到 CPU（内存 ~"
          f"{round(sum(t.numel() * t.element_size() for t in state.values()) / 2 ** 30, 1)} GB）")

    # 3) merge
    merged_records, unmatched = merge_pairs(state, pairs, scaling)
    if unmatched and args.strict:
        raise RuntimeError(
            f"存在 {len(unmatched)} 个未匹配 adapter 键（--strict）:\n"
            + "\n".join(f"  - {u['module']}: {u['reason']}" for u in unmatched)
        )
    if unmatched:
        print(f"[警告] {len(unmatched)} 个 adapter 键未匹配基座（已写入报告）:")
        for u in unmatched:
            print(f"  - {u['module']}: {u['reason']}")

    # 4) 输出目录防护
    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir) and not args.force:
        raise FileExistsError(
            f"输出目录 {args.output_dir} 非空（防误覆盖；确认可清空后加 --force）"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    merged_base_keys = {r["base_key"] for r in merged_records}
    rewrites, copies = write_merged_output(
        args.base_dir, args.output_dir, files, state, merged_base_keys
    )
    del state

    # 5) 报告
    report = {
        "adapter": os.path.abspath(args.adapter),
        "adapter_format": payload.get("format"),
        "adapter_config": config,
        "scaling": scaling,
        "summary": {
            "paired_modules": len(pairs),
            "expected_pairs": expected,
            "merged_keys": len(merged_records),
            "unmatched_adapter_keys_in_payload": len(unmatched_keys),
            "unmatched_pairs": len(unmatched),
        },
        "merged": merged_records,
        "unmatched": unmatched + [{"key": k, "reason": "state_dict 键无法归一到 blocks.* 目标"}
                                  for k in unmatched_keys],
        "shard_rewrites": rewrites,
        "shard_copies": copies,
        "base_dir": os.path.abspath(args.base_dir),
        "output_dir": os.path.abspath(args.output_dir),
    }
    report_path = os.path.join(args.output_dir, "merge_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[报告] {report_path}")
    print(f"[合并完成] 覆盖 {len(merged_records)} 个基座键；"
          f"重写 {len(rewrites)} 个分片 / 复制 {len(copies)} 个分片")

    # 6) 可选桥转换
    if args.bridge_output_dir:
        if os.path.isdir(args.bridge_output_dir) and os.listdir(args.bridge_output_dir) \
                and not args.force:
            raise FileExistsError(
                f"桥输出目录 {args.bridge_output_dir} 非空（防误覆盖；确认可清空后加 --force）"
            )
        run_bridge(args.output_dir, args.bridge_output_dir, args.bridge_dtype,
                   args.max_shard_size_gb, args.strict, args.check_hash)
        print(f"[桥转换完成] DiffSynth 擦除后基座 → {args.bridge_output_dir}")
    else:
        print_bridge_command(args.output_dir, None, args.bridge_dtype,
                             args.max_shard_size_gb, args.strict, args.check_hash)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Wan2.2-TI2V-5B 擦除 LoRA 合并（VU adapter_final.pt → diffusers "
                    "transformer）→ Task1 桥转 DiffSynth 基座",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法")[1] if "用法" in __doc__ else None,
    )
    parser.add_argument("--adapter", type=str, required=True,
                        help="VU GradAscent 适配器：adapter_final.pt 或 adapter_step<N>.pt")
    parser.add_argument("--base-dir", type=str, required=True,
                        help="diffusers Wan2.2-TI2V-5B transformer 目录"
                             "（含 diffusion_pytorch_model*.safetensors + config.json）")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="合并后 diffusers transformer 目录"
                             "（落 models/unlearn/wan5b_nudity_grad_ascent/merged/）")
    parser.add_argument("--bridge-output-dir", type=str, default=None,
                        help="桥转换输出目录（DiffSynth 擦除后基座，落 "
                             "models/Wan-AI/Wan2.2-TI2V-5B-erased/）；缺省则只合并不转桥")
    parser.add_argument("--bridge-dtype", type=str, default="bf16",
                        choices=["auto", "bf16", "fp16", "fp32"],
                        help="桥转换输出 dtype（合并侧保持基座 dtype）")
    parser.add_argument("--max-shard-size-gb", type=float, default=4.0,
                        help="桥转换分片大小（GB），参考 1.3B 布局")
    parser.add_argument("--strict", action="store_true",
                        help="adapter 键未匹配基座即失败（默认警告并写入报告）")
    parser.add_argument("--check-hash", action="store_true",
                        help="桥转换后校验 DiffSynth 注册表 key+shape 哈希")
    parser.add_argument("--force", action="store_true",
                        help="输出/桥输出目录非空时允许覆盖")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印合并+桥转换计划，不读 tensor、不写文件")
    args = parser.parse_args()

    rc = run_dry_run(args) if args.dry_run else run_merge(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()