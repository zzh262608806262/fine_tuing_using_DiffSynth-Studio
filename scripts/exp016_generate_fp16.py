#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp016 Stage 0: 生成FP16基线视频（纯擦除模型，无LoRA）

5个模型：
    baseline            = Wan-AI/Wan2.2-TI2V-5B（原始）
    esd_erased          = wan5b/exp015_esd_erased
    npo_erased          = wan5b/exp015_npo_erased
    grad_ascent_erased  = wan5b/exp015_grad_ascent_erased
    anchor_distill_erased = wan5b/exp015_anchor_distill_erased

评测集：nudity domain 99条（复用 Exp015 的 benchmark）

输出：outputs/exp016_baseline/<model>/<idx:03d>.mp4

用法：
    # 预览
    python3 scripts/exp016_generate_fp16.py --model baseline --dry-run

    # 生成（集群）
    python -u scripts/exp016_generate_fp16.py --model baseline --start 0 --end 99
"""
import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ---------------- 常量 ----------------
VU_ROOT = os.getenv("VU_ROOT", "../video-unlearning")

# 5个模型定义
MODELS = {
    "baseline": {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "description": "Original Wan2.2-TI2V-5B (baseline)",
    },
    "esd_erased": {
        "model_id": "wan5b/exp015_esd_erased",
        "description": "ESD erased (pure, no LoRA)",
    },
    "npo_erased": {
        "model_id": "wan5b/exp015_npo_erased",
        "description": "NPO erased (pure, no LoRA)",
    },
    "grad_ascent_erased": {
        "model_id": "wan5b/exp015_grad_ascent_erased",
        "description": "GradAscent erased (pure, no LoRA)",
    },
    "anchor_distill_erased": {
        "model_id": "wan5b/exp015_anchor_distill_erased",
        "description": "AnchorDistill erased (pure, no LoRA)",
    },
}

# 生成参数
GEN_PARAMS = {
    "height": 480,
    "width": 736,
    "num_frames": 17,
    "num_inference_steps": 50,
    "cfg_scale": 7.0,
    "seed": 42,
}


def load_prompts():
    """加载 VU benchmark 的 nudity prompts"""
    benchmark_file = Path(VU_ROOT) / "data" / "benchmark_wan.jsonl"
    if not benchmark_file.exists():
        raise FileNotFoundError(f"Benchmark not found: {benchmark_file}")

    prompts = []
    with open(benchmark_file) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("domain") == "nudity":
                prompts.append(entry["prompt"])

    print(f"✓ 加载了 {len(prompts)} 条 nudity prompts")
    return prompts


def build_pipeline(model_name):
    """构建 FP16 pipeline（无LoRA）"""
    import torch
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

    model_config = MODELS[model_name]
    model_id = model_config["model_id"]

    print(f"\n{'='*70}")
    print(f"加载模型: {model_name}")
    print(f"  描述: {model_config['description']}")
    print(f"  模型ID: {model_id}")
    print(f"{'='*70}\n")

    # 模型文件列表
    MODEL_FILES = [
        "diffusion_pytorch_model*.safetensors",
        "models_t5_umt5-xxl-enc-bf16.safetensors",
        "Wan2.2_VAE.safetensors",
    ]

    # 构建配置
    print("加载模型...")
    model_configs = [
        ModelConfig(model_id=model_id, origin_file_pattern=pattern)
        for pattern in MODEL_FILES
    ]

    tokenizer_config = ModelConfig(
        model_id="Wan-AI/Wan2.1-T2V-1.3B",
        origin_file_pattern="google/umt5-xxl/"
    )

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )

    print("✓ Pipeline 构建完成\n")
    return pipe


def generate_videos(pipe, model_name, prompts, start_idx, end_idx, output_dir, dry_run=False):
    """生成视频"""
    from diffsynth.utils.data import save_video
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存生成参数
    params_file = output_dir / "gen_params.json"
    with open(params_file, "w") as f:
        json.dump({
            "model": model_name,
            "model_config": MODELS[model_name],
            "gen_params": GEN_PARAMS,
            "num_prompts": len(prompts),
        }, f, indent=2)

    if dry_run:
        print(f"[DRY RUN] 将生成 {end_idx - start_idx} 个视频")
        print(f"输出目录: {output_dir}")
        return

    print(f"\n{'='*70}")
    print(f"开始生成: {model_name}")
    print(f"  范围: [{start_idx}, {end_idx})")
    print(f"  输出: {output_dir}")
    print(f"{'='*70}\n")

    for idx in range(start_idx, end_idx):
        if idx >= len(prompts):
            print(f"⚠️  索引 {idx} 超出范围，跳过")
            continue

        prompt = prompts[idx]
        output_path = output_dir / f"{idx:03d}.mp4"

        # 跳过已存在的
        if output_path.exists():
            print(f"[{idx:03d}] 已存在，跳过")
            continue

        print(f"[{idx:03d}] {prompt[:60]}...")

        try:
            video = pipe(
                prompt=prompt,
                height=GEN_PARAMS["height"],
                width=GEN_PARAMS["width"],
                num_frames=GEN_PARAMS["num_frames"],
                num_inference_steps=GEN_PARAMS["num_inference_steps"],
                cfg_scale=GEN_PARAMS["cfg_scale"],
                seed=GEN_PARAMS["seed"],
            )
            save_video(video, str(output_path), fps=24, quality=5)
            print(f"  ✓ 保存到: {output_path}")

        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ {model_name} 生成完成\n")


def main():
    parser = argparse.ArgumentParser(description="Exp016 Stage 0: FP16基线生成")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()),
                        help="模型名称")
    parser.add_argument("--start", type=int, default=0,
                        help="起始索引（默认: 0）")
    parser.add_argument("--end", type=int, default=99,
                        help="结束索引（默认: 99）")
    parser.add_argument("--output-dir", type=str,
                        default=None,
                        help="输出目录（默认: outputs/exp016_baseline/<model>）")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式，不实际生成")

    args = parser.parse_args()

    # 确定输出目录
    if args.output_dir is None:
        args.output_dir = f"outputs/exp016_baseline/{args.model}"

    print(f"\n生成配置:")
    print(f"  模型: {args.model}")
    print(f"  提示词范围: [{args.start}, {args.end})")
    print(f"  输出: {args.output_dir}")
    print()

    # 加载prompts
    prompts = load_prompts()

    # 构建pipeline
    pipe = build_pipeline(args.model)

    # 生成视频
    generate_videos(
        pipe=pipe,
        model_name=args.model,
        prompts=prompts,
        start_idx=args.start,
        end_idx=args.end,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
