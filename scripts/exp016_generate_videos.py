#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp016 量化模型视频生成（8个量化臂）

8臂定义（量化版本）：
    esd_erased_nf4         = quantized exp015_esd_erased + LoRA
    esd_base_nf4          = quantized Wan2.2-TI2V-5B + LoRA
    npo_erased_nf4        = quantized exp015_npo_erased + LoRA
    npo_base_nf4          = quantized Wan2.2-TI2V-5B + LoRA
    grad_ascent_erased_nf4 = quantized exp015_grad_ascent_erased + LoRA
    grad_ascent_base_nf4   = quantized Wan2.2-TI2V-5B + LoRA
    anchor_distill_erased_nf4 = quantized exp015_anchor_distill_erased + LoRA
    anchor_distill_base_nf4   = quantized Wan2.2-TI2V-5B + LoRA

评测集：nudity domain 99条（复用 Exp015 的 benchmark）

输出：outputs/exp016/<arm>/<idx:03d>.mp4 + gen_params.json

用法：
    # 预览
    python3 scripts/exp016_generate_videos.py --arm esd_erased_nf4 --dry-run

    # 生成（集群）
    python -u scripts/exp016_generate_videos.py --arm esd_erased_nf4 --start 0 --end 99
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

# 8个量化臂定义
ARMS = {
    "esd_erased_nf4": {
        "quantized_model": "models/quantized/exp016_esd_erased_nf4",
        "description": "ESD erased (quantized NF4)",
    },
    "esd_base_nf4": {
        "quantized_model": "models/quantized/exp016_esd_base_nf4",
        "description": "ESD base (quantized NF4)",
    },
    "npo_erased_nf4": {
        "quantized_model": "models/quantized/exp016_npo_erased_nf4",
        "description": "NPO erased (quantized NF4)",
    },
    "npo_base_nf4": {
        "quantized_model": "models/quantized/exp016_npo_base_nf4",
        "description": "NPO base (quantized NF4)",
    },
    "grad_ascent_erased_nf4": {
        "quantized_model": "models/quantized/exp016_grad_ascent_erased_nf4",
        "description": "GradAscent erased (quantized NF4)",
    },
    "grad_ascent_base_nf4": {
        "quantized_model": "models/quantized/exp016_grad_ascent_base_nf4",
        "description": "GradAscent base (quantized NF4)",
    },
    "anchor_distill_erased_nf4": {
        "quantized_model": "models/quantized/exp016_anchor_distill_erased_nf4",
        "description": "AnchorDistill erased (quantized NF4)",
    },
    "anchor_distill_base_nf4": {
        "quantized_model": "models/quantized/exp016_anchor_distill_base_nf4",
        "description": "AnchorDistill base (quantized NF4)",
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


def build_pipeline(arm):
    """
    构建量化模型的 pipeline

    策略：先加载原始模型，再加载量化权重并应用量化
    """
    import torch
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.core.quant import QuantizeConfig
    from safetensors.torch import load_file

    arm_config = ARMS[arm]
    quantized_model_dir = Path(arm_config["quantized_model"])
    quantized_dit_path = quantized_model_dir / "diffusion_pytorch_model.safetensors"

    if not quantized_dit_path.exists():
        raise FileNotFoundError(f"量化模型不存在: {quantized_dit_path}")

    print(f"\n{'='*70}")
    print(f"加载量化模型: {arm}")
    print(f"  描述: {arm_config['description']}")
    print(f"  量化DiT: {quantized_dit_path}")
    print(f"{'='*70}\n")

    # 获取原始基座（用于确定模型ID）
    orig_arm = arm.replace("_nf4", "")
    if "erased" in arm:
        base_model_id = f"wan5b/exp015_{orig_arm}"
    else:
        base_model_id = "Wan-AI/Wan2.2-TI2V-5B"

    # 模型文件列表
    MODEL_FILES = [
        "diffusion_pytorch_model*.safetensors",
        "models_t5_umt5-xxl-enc-bf16.safetensors",
        "Wan2.2_VAE.safetensors",
    ]

    # 构建配置（先加载原始结构）
    print("加载原始模型结构...")
    model_configs = [
        ModelConfig(model_id=base_model_id, origin_file_pattern=pattern)
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

    # 应用量化（创建量化结构）
    print("应用量化配置...")
    quant_config = QuantizeConfig(method="bitsandbytes_nf4")
    quant_config.quantize_model(pipe.dit, compute_device="cuda", model_device="cuda")

    # 加载量化权重
    print(f"加载量化权重: {quantized_dit_path}")
    quantized_state = load_file(str(quantized_dit_path))
    pipe.dit.load_state_dict(quantized_state, strict=False)

    print("✓ Pipeline 构建完成\n")
    return pipe


def generate_videos(pipe, arm, prompts, start_idx, end_idx, output_dir, dry_run=False):
    """生成视频"""
    from diffsynth.utils.data import save_video
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存生成参数
    params_file = output_dir / "gen_params.json"
    with open(params_file, "w") as f:
        json.dump({
            "arm": arm,
            "arm_config": ARMS[arm],
            "gen_params": GEN_PARAMS,
            "num_prompts": len(prompts),
        }, f, indent=2)

    if dry_run:
        print(f"[DRY RUN] 将生成 {end_idx - start_idx} 个视频")
        print(f"输出目录: {output_dir}")
        return

    print(f"\n{'='*70}")
    print(f"开始生成: {arm}")
    print(f"  范围: [{start_idx}, {end_idx})")
    print(f"  输出: {output_dir}")
    print(f"{'='*70}\n")

    for idx in range(start_idx, end_idx):
        if idx >= len(prompts):
            print(f"⚠️  索引 {idx} 超出范围，跳过")
            continue

        prompt = prompts[idx]
        output_path = output_dir / f"{idx:03d}.mp4"

        if output_path.exists():
            print(f"[{idx:03d}] 已存在，跳过")
            continue

        print(f"[{idx:03d}] {prompt[:60]}...")

        try:
            with torch.no_grad():
                video = pipe(
                    prompt=prompt,
                    negative_prompt="",
                    height=GEN_PARAMS["height"],
                    width=GEN_PARAMS["width"],
                    num_frames=GEN_PARAMS["num_frames"],
                    num_inference_steps=GEN_PARAMS["num_inference_steps"],
                    cfg_scale=GEN_PARAMS["cfg_scale"],
                    seed=GEN_PARAMS["seed"],
                )

            save_video(video, output_path, fps=24, quality=5)
            print(f"  ✓ 保存到: {output_path}")

        except Exception as e:
            print(f"  ✗ 失败: {e}")
            continue

    print(f"\n✅ 生成完成: {arm}")


def main():
    parser = argparse.ArgumentParser(description="Exp016 量化模型视频生成")
    parser.add_argument("--arm", required=True, choices=list(ARMS.keys()),
                        help="量化模型臂")
    parser.add_argument("--start", type=int, default=0,
                        help="起始索引")
    parser.add_argument("--end", type=int, default=99,
                        help="结束索引（不含）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录（默认: outputs/exp016/<arm>）")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式，不实际生成")

    args = parser.parse_args()

    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = f"outputs/exp016/{args.arm}"

    # 加载 prompts
    prompts = load_prompts()

    # 构建 pipeline
    if not args.dry_run:
        pipe = build_pipeline(args.arm)
    else:
        pipe = None

    # 生成视频
    generate_videos(
        pipe, args.arm, prompts,
        args.start, args.end,
        args.output_dir, args.dry_run
    )


if __name__ == "__main__":
    main()
