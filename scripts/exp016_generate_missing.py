#!/usr/bin/env python3
"""
Exp016: 重新生成缺失的视频（000-023）
用于补充v4批次被覆盖后缺失的视频
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到路径
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# 臂配置（复用exp016_generate_videos.py的配置）
ARM_CONFIGS = {
    "esd_erased_nf4": {
        "quantized_model": "models/quantized/exp016_esd_erased_nf4",
        "description": "ESD erased (quantized NF4)",
        "missing_range": (0, 24),  # 000-023
    },
    "esd_base_nf4": {
        "quantized_model": "models/quantized/exp016_esd_base_nf4",
        "description": "ESD base (quantized NF4)",
        "missing_range": (0, 23),  # 000-022
    },
    "npo_erased_nf4": {
        "quantized_model": "models/quantized/exp016_npo_erased_nf4",
        "description": "NPO erased (quantized NF4)",
        "missing_range": (0, 23),
    },
    "npo_base_nf4": {
        "quantized_model": "models/quantized/exp016_npo_base_nf4",
        "description": "NPO base (quantized NF4)",
        "missing_range": (0, 23),
    },
    "grad_ascent_erased_nf4": {
        "quantized_model": "models/quantized/exp016_grad_ascent_erased_nf4",
        "description": "GradAscent erased (quantized NF4)",
        "missing_range": (0, 24),
    },
    "grad_ascent_base_nf4": {
        "quantized_model": "models/quantized/exp016_grad_ascent_base_nf4",
        "description": "GradAscent base (quantized NF4)",
        "missing_range": (0, 24),
    },
    "anchor_distill_erased_nf4": {
        "quantized_model": "models/quantized/exp016_anchor_distill_erased_nf4",
        "description": "AnchorDistill erased (quantized NF4)",
        "missing_range": (0, 24),
    },
    "anchor_distill_base_nf4": {
        "quantized_model": "models/quantized/exp016_anchor_distill_base_nf4",
        "description": "AnchorDistill base (quantized NF4)",
        "missing_range": (0, 24),
    },
}

# 生成参数（与exp016_generate_videos.py一致）
GEN_PARAMS = {
    "height": 480,
    "width": 736,
    "num_frames": 17,
    "num_inference_steps": 50,
    "cfg_scale": 7.0,
    "seed": 42,
}

# VU项目根目录
VU_ROOT = os.getenv("VU_ROOT", "../video-unlearning")


def load_prompts():
    """加载VU benchmark的nudity prompts（复用exp016_generate_videos.py逻辑）"""
    benchmark_file = Path(VU_ROOT) / "data" / "benchmark_wan.jsonl"
    if not benchmark_file.exists():
        raise FileNotFoundError(f"Benchmark not found: {benchmark_file}")

    prompts = []
    with open(benchmark_file) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("domain") == "nudity":
                prompts.append(entry["prompt"])

    if len(prompts) != 99:
        raise ValueError(f"Expected 99 nudity prompts, got {len(prompts)}")

    return prompts


def build_pipeline(arm_name):
    """构建量化模型的pipeline（复用exp016_generate_videos.py的逻辑）"""
    import torch
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.core.quant import QuantizeConfig
    from safetensors.torch import load_file

    config = ARM_CONFIGS[arm_name]
    quantized_model_dir = Path(config["quantized_model"])
    quantized_dit_path = quantized_model_dir / "diffusion_pytorch_model.safetensors"

    if not quantized_dit_path.exists():
        raise FileNotFoundError(f"量化模型不存在: {quantized_dit_path}")

    print(f"\n{'='*70}")
    print(f"加载量化模型: {arm_name}")
    print(f"  描述: {config['description']}")
    print(f"  量化DiT: {quantized_dit_path}")
    print(f"{'='*70}\n")

    # 获取原始基座
    orig_arm = arm_name.replace("_nf4", "")
    if "erased" in arm_name:
        base_model_id = f"wan5b/exp015_{orig_arm}"
    else:
        base_model_id = "Wan-AI/Wan2.2-TI2V-5B"

    # 模型文件列表
    MODEL_FILES = [
        "diffusion_pytorch_model*.safetensors",
        "models_t5_umt5-xxl-enc-bf16.safetensors",
        "Wan2.2_VAE.safetensors",
    ]

    # 构建配置
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


def generate_missing_videos(arm_name):
    """为指定臂生成缺失的视频"""
    from diffsynth.utils.data import save_video
    import torch

    if arm_name not in ARM_CONFIGS:
        raise ValueError(f"未知臂: {arm_name}")

    config = ARM_CONFIGS[arm_name]
    output_dir = Path(f"outputs/exp016/{arm_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载prompts
    all_prompts = load_prompts()
    start_idx, end_idx = config["missing_range"]
    missing_prompts = all_prompts[start_idx:end_idx]

    print(f"{'='*70}")
    print(f"生成缺失视频: {arm_name}")
    print(f"{'='*70}")
    print(f"配置: {config['description']}")
    print(f"量化模型: {config['quantized_model']}")
    print(f"缺失范围: {start_idx:03d} - {end_idx-1:03d} (共 {len(missing_prompts)} 个)")
    print(f"输出目录: {output_dir}")
    print(f"生成参数: {GEN_PARAMS}")
    print()

    # 加载模型
    pipe = build_pipeline(arm_name)

    # 生成视频
    for i, prompt in enumerate(missing_prompts):
        video_idx = start_idx + i
        output_path = output_dir / f"{video_idx:03d}.mp4"

        # 跳过已存在的
        if output_path.exists():
            print(f"[{video_idx:03d}] 已存在，跳过")
            continue

        print(f"[{video_idx:03d}] {prompt[:60]}...")

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
            continue

    print()
    print(f"✅ 补充生成完成: {arm_name}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp016: 补充生成缺失视频")
    parser.add_argument("--arm", required=True, choices=list(ARM_CONFIGS.keys()),
                        help="要生成的臂")

    args = parser.parse_args()

    generate_missing_videos(args.arm)
