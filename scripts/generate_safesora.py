"""
SafeSora 测试集视频批量生成 — 三种方法对比 (lora / distill / quant)

- 测试集: data/safesora/config-test.json.gz 的 1471 个唯一 prompt（按 prompt_id 排序）
- 输出:   outputs/safesora_gen/<method>/<prompt_id>.mp4
- 断点续跑: 已存在的 mp4 自动跳过；先写 .tmp.mp4 再原子重命名，避免半截文件被当成完成
- 分片:   --shard i --num_shards n 按 index % n == i 划分，多卡并行

用法:
    python scripts/generate_safesora.py --method lora --shard 0 --num_shards 2
    python scripts/generate_safesora.py --method distill
    python scripts/generate_safesora.py --method quant --limit 1   # 冒烟测试

方法配置:
    base    = 基座 Wan2.1-T2V-1.3B（微调前对照组）,     30 步, cfg 5.0
    lora    = 基座 + lora_70/epoch-1 (alpha=1),        30 步, cfg 5.0
    distill = 基座 + distill_4step/epoch-4 (alpha=1),   4 步, cfg 1.0
    quant   = nf4 量化 DiT checkpoint,                 30 步, cfg 5.0

抽样 (--sample N): 只取不安全提示词（prompt_type == safety_critical，测试集共 745 条唯一，
全部带不安全标签），用 md5(seed:prompt_id) 排序取前 N —— 确定性、且扩大 N 时
旧样本仍在其中（前缀稳定）。--sample 0 表示全部 745 条 critical。
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch

from quantize import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_NEGATIVE,
    QuantizeConfig,
    load_pipeline,
)

METHOD_CFG = {
    "base": {
        "ckpt": None,
        "steps": 30, "cfg": 5.0,
    },
    "lora": {
        "ckpt": "./models/train/Wan2.1-T2V-1.3B_lora_70/epoch-1.safetensors",
        "steps": 30, "cfg": 5.0,
    },
    "distill": {
        "ckpt": "./models/train/Wan2.1-T2V-1.3B_distill_4step/epoch-4.safetensors",
        "steps": 4, "cfg": 1.0,
    },
    "quant": {
        "ckpt": "./models/quantized/Wan2.1-T2V-1.3B_nf4.safetensors",
        "steps": 30, "cfg": 5.0,
    },
    # Exp008 恶意微调 N=3 投毒 LoRA（与 lora 同为 base+LoRA alpha=1，参数对齐保证可比）
    "malicious": {
        "ckpt": "./models/train/Wan2.1-T2V-1.3B_lora_malicious_N3/epoch-4.safetensors",
        "steps": 30, "cfg": 5.0,
    },
}


def load_test_prompts(path):
    with gzip.open(path, "rt") as f:
        entries = json.load(f)
    seen = {}
    for e in entries:
        pid = e["prompt_id"]
        if pid not in seen:
            seen[pid] = {
                "prompt_id": pid,
                "prompt_text": e["prompt_text"].strip(),
                "prompt_type": e["prompt_type"],
                "prompt_labels": e.get("prompt_labels"),
            }
    return sorted(seen.values(), key=lambda x: x["prompt_id"])


def sample_unsafe(prompts, n, seed):
    """只取 safety_critical（不安全）提示词, md5 哈希排序保证确定性和前缀稳定"""
    unsafe = [p for p in prompts if p["prompt_type"] == "safety_critical"]
    if n:
        rank = lambda p: hashlib.md5(f"{seed}:{p['prompt_id']}".encode()).hexdigest()
        unsafe = sorted(unsafe, key=rank)[:n]
    return sorted(unsafe, key=lambda x: x["prompt_id"])


def build_pipe(method):
    cfg = METHOD_CFG[method]
    if method in ("base", "lora", "distill", "malicious"):
        pipe = load_pipeline(DEFAULT_MODEL_ID, device="cuda")
        if cfg["ckpt"]:
            print(f"加载 LoRA: {cfg['ckpt']}")
            pipe.load_lora(pipe.dit, cfg["ckpt"], alpha=1)
        return pipe

    # quant: 与 quantize.py mode_load 相同的加载流程
    import importlib
    from safetensors import safe_open

    quantized_path = cfg["ckpt"]
    config_path = quantized_path.rsplit(".", 1)[0] + ".config.json"
    with open(config_path) as f:
        config = json.load(f)

    pipe = load_pipeline(DEFAULT_MODEL_ID, device="cuda")
    print(f"加载量化 DiT: {quantized_path}")
    module_path, class_name = config["model_class"].rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_path), class_name)
    dit = model_class(**config["extra_kwargs"])
    quant_config = QuantizeConfig(**config["quant_config"])
    quant_config.prepare_for_prequantized_load(dit, compute_dtype=torch.bfloat16)

    state_dict, metadata = {}, {}
    with safe_open(quantized_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
        metadata = f.metadata() or {}
    state_dict = quant_config.unflatten_state_dict(state_dict, metadata)
    dit.load_state_dict(state_dict, assign=True)
    dit = dit.to(device="cuda", dtype=torch.bfloat16)
    del pipe.dit
    pipe.dit = dit
    return pipe


def main():
    parser = argparse.ArgumentParser(description="SafeSora 测试集三方法批量生成")
    parser.add_argument("--method", choices=list(METHOD_CFG), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="只生成前 N 个（冒烟测试）")
    parser.add_argument("--test_config", type=str, default="./data/safesora/config-test.json.gz")
    parser.add_argument("--output_root", type=str, default="./outputs/safesora_gen")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", type=int, default=200,
                        help="从 safety_critical 中抽的条数；0=全部 745 条 critical")
    parser.add_argument("--sample_seed", type=int, default=0)
    args = parser.parse_args()

    from diffsynth.utils.data import save_video

    prompts = load_test_prompts(args.test_config)
    prompts = sample_unsafe(prompts, args.sample, args.sample_seed)
    os.makedirs(args.output_root, exist_ok=True)
    manifest_name = f"prompts_unsafe_{args.sample or 'all'}.json"
    manifest_path = os.path.join(args.output_root, manifest_name)
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=1)

    todo = [p for i, p in enumerate(prompts) if i % args.num_shards == args.shard]
    # 全局安全红线：绝不生成 child_abuse 标注的 prompt（CSAM），对所有方法生效
    _before = len(todo)
    todo = [p for p in todo if not (p.get("prompt_labels") or {}).get("child_abuse")]
    if _before != len(todo):
        print(f"[safety] 跳过 child_abuse prompt {_before - len(todo)} 条，实际生成 {len(todo)}")
    if args.limit:
        todo = todo[: args.limit]
    out_dir = os.path.join(args.output_root, args.method)
    os.makedirs(out_dir, exist_ok=True)

    remaining = [p for p in todo if not os.path.exists(os.path.join(out_dir, p["prompt_id"] + ".mp4"))]
    print(f"[{args.method}] shard {args.shard}/{args.num_shards}: "
          f"共 {len(todo)}，已完成 {len(todo) - len(remaining)}，待生成 {len(remaining)}")
    if not remaining:
        print("SHARD_COMPLETE")
        return

    cfg = METHOD_CFG[args.method]
    pipe = build_pipe(args.method)

    for n, p in enumerate(remaining, 1):
        out_path = os.path.join(out_dir, p["prompt_id"] + ".mp4")
        if os.path.exists(out_path):  # 并行分片间保险
            continue
        t0 = time.time()
        video = pipe(
            prompt=p["prompt_text"],
            negative_prompt=DEFAULT_NEGATIVE if cfg["cfg"] > 1 else "",
            seed=args.seed,
            height=480, width=832, num_frames=81,
            num_inference_steps=cfg["steps"],
            cfg_scale=cfg["cfg"],
            tiled=True,
        )
        tmp_path = out_path + ".tmp.mp4"
        save_video(video, tmp_path, fps=15, quality=5)
        os.replace(tmp_path, out_path)
        print(f"[{args.method}] {n}/{len(remaining)} {p['prompt_id'][:12]} "
              f"({time.time() - t0:.0f}s)", flush=True)

    print("SHARD_COMPLETE")


if __name__ == "__main__":
    main()
