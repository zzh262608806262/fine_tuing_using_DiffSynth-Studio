#!/usr/bin/env python3
"""
Exp020a: 生成NPOMasked擦除后的视频
直接加载合并后的模型，不需要LoRA
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


def load_prompts(prompts_file):
    """加载评测prompts"""
    prompts = []
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data['prompt'])
    return prompts


def generate_videos(args):
    """生成视频"""
    print("=" * 60)
    print(f"加载模型: {args.model_dir}")
    print("=" * 60)

    # 加载模型 - Diffusers格式：直接传目录
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.float16,
        device="cuda",
        model_configs=[
            ModelConfig(path=args.model_dir),  # DiT (自动读取model_index.json)
            ModelConfig(path=f"{args.model_dir}/models_t5_umt5-xxl-enc-bf16.safetensors"),
            ModelConfig(path=f"{args.model_dir}/Wan2.2_VAE.safetensors"),
        ],
    )
    print("✅ 模型加载完成")

    # 加载prompts
    prompts = load_prompts(args.prompts_file)
    print(f"\n加载了 {len(prompts)} 条prompts")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成视频
    print(f"\n开始生成 (frames={args.num_frames}, size={args.height}×{args.width}, steps={args.num_inference_steps})")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    success_count = 0
    for i, prompt in enumerate(tqdm(prompts, desc="生成视频")):
        output_path = output_dir / f"video_{i:03d}.mp4"

        # 跳过已存在的
        if output_path.exists():
            print(f"  跳过 {i:03d} (已存在)")
            success_count += 1
            continue

        try:
            video = pipe(
                prompt=prompt,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
            )

            # 保存
            pipe.save_video(video, str(output_path), fps=args.fps)
            success_count += 1

        except Exception as e:
            print(f"  ❌ 生成失败 {i:03d}: {e}")
            continue

    print("=" * 60)
    print(f"✅ 完成: {success_count}/{len(prompts)} 个视频")
    print("=" * 60)

    return success_count == len(prompts)


def main():
    parser = argparse.ArgumentParser(description="Exp020a视频生成")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="合并后的模型目录")
    parser.add_argument("--prompts-file", type=str, required=True,
                        help="prompts文件路径 (.jsonl)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出视频目录")
    parser.add_argument("--num-frames", type=int, default=17,
                        help="视频帧数")
    parser.add_argument("--height", type=int, default=480,
                        help="视频高度")
    parser.add_argument("--width", type=int, default=736,
                        help="视频宽度")
    parser.add_argument("--num-inference-steps", type=int, default=50,
                        help="推理步数")
    parser.add_argument("--guidance-scale", type=float, default=5.0,
                        help="CFG scale")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子")
    parser.add_argument("--fps", type=int, default=8,
                        help="视频帧率")

    args = parser.parse_args()

    # 检查输入
    if not os.path.isdir(args.model_dir):
        print(f"❌ 模型目录不存在: {args.model_dir}")
        sys.exit(1)

    if not os.path.isfile(args.prompts_file):
        print(f"❌ Prompts文件不存在: {args.prompts_file}")
        sys.exit(1)

    # 生成
    success = generate_videos(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
