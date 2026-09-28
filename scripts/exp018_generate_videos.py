#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp018 视频生成脚本 - 加载擦除基座 + 微调 LoRA epoch-19

用法:
    python -u scripts/exp018_generate_videos.py \
        --method esd \
        --base_model models/wan5b/exp015_esd_erased \
        --lora_path models/finetune/exp018_esd_ft/epoch-19.safetensors \
        --output_dir outputs/exp018/esd_ft_e19
"""
import argparse
import json
import os
import sys
import time

# 添加项目路径
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

VU_ROOT = os.getenv("VU_ROOT", "../video-unlearning")


def load_benchmark_prompts():
    """加载 nudity 评测 prompts (99条) - 复用 Exp015 的文件"""
    prompts_file = "data/exp015/nudity_eval_prompts.jsonl"

    if not os.path.isfile(prompts_file):
        raise FileNotFoundError(f"评测 prompts 文件不存在: {prompts_file}")

    prompts = []
    with open(prompts_file, "r") as f:
        for line in f:
            item = json.loads(line)
            prompts.append({
                "prompt": item["prompt"],
                "seed": item.get("seed", 42),
            })
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, help="方法名: esd, npo, grad_ascent, anchor_distill")
    parser.add_argument("--base_model", required=True, help="擦除基座路径")
    parser.add_argument("--lora_path", required=True, help="微调 LoRA 路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--flow_shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    print(f"========================================")
    print(f"Exp018 视频生成: {args.method}")
    print(f"========================================")
    print(f"基座: {args.base_model}")
    print(f"LoRA: {args.lora_path}")
    print(f"输出: {args.output_dir}")
    print(f"")

    # 加载 prompts
    prompts = load_benchmark_prompts()
    print(f"✓ 加载 {len(prompts)} 条 nudity prompts")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存生成参数
    gen_params = {
        "method": args.method,
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "num_frames": args.frames,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.cfg,
        "flow_shift": args.flow_shift,
        "fps": args.fps,
        "total_prompts": len(prompts),
    }

    with open(os.path.join(args.output_dir, "gen_params.json"), "w") as f:
        json.dump(gen_params, f, indent=2)

    print(f"✓ 保存生成参数")
    print(f"")

    # 初始化 pipeline
    print(f"[1/2] 加载模型...")

    import torch
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.utils.data import save_video

    # 模型文件 patterns
    MODEL_FILES = [
        "diffusion_pytorch_model*.safetensors",
        "models_t5_umt5-xxl-enc-bf16.safetensors",
        "Wan2.2_VAE.safetensors",
    ]

    # 从完整路径提取 model_id (相对于 models/ 目录)
    # 例如: models/wan5b/exp015_esd_erased -> wan5b/exp015_esd_erased
    if args.base_model.startswith("models/"):
        model_id = args.base_model[len("models/"):]
    else:
        model_id = args.base_model

    # 构建模型配置
    model_configs = [
        ModelConfig(model_id=model_id, origin_file_pattern=pattern)
        for pattern in MODEL_FILES
    ]

    # Tokenizer 配置
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

    # 加载微调 LoRA
    print(f"  加载 LoRA: {args.lora_path}")
    pipe.load_lora(pipe.dit, args.lora_path, alpha=1.0)

    print(f"✓ Pipeline 就绪")
    print(f"")

    # 生成视频
    print(f"[2/2] 开始生成 {len(prompts)} 条视频...")

    for idx, prompt_data in enumerate(prompts):
        prompt = prompt_data["prompt"]
        seed = prompt_data["seed"]
        output_path = os.path.join(args.output_dir, f"{idx:03d}.mp4")

        if os.path.exists(output_path):
            print(f"  [{idx:03d}/{len(prompts):03d}] 跳过（已存在）")
            continue

        print(f"  [{idx:03d}/{len(prompts):03d}] {prompt[:60]}... (seed={seed})")

        try:
            start_time = time.time()

            video = pipe(
                prompt=prompt,
                negative_prompt="",
                seed=seed,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                num_inference_steps=args.steps,
                cfg_scale=args.cfg,
                sigma_shift=args.flow_shift,
                tiled=True,
            )

            save_video(video, output_path, fps=args.fps, quality=5)
            elapsed = time.time() - start_time

            print(f"    ✓ 保存: {output_path} ({elapsed:.1f}s)")

        except Exception as e:
            print(f"    ⚠️  生成失败: {e}")
            continue

    print(f"")
    print(f"========================================")
    print(f"✅ 生成完成")

    video_count = len([f for f in os.listdir(args.output_dir) if f.endswith(".mp4")])
    print(f"  总计: {video_count}/{len(prompts)} 个视频")
    print(f"  输出: {args.output_dir}")
    print(f"========================================")


if __name__ == "__main__":
    main()
