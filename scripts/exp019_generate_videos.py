#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp019 视频生成（GradDiff对照实验）

臂定义：
    graddiff_base   = Wan2.2-TI2V-5B（原始基座，无擦除）
    graddiff_erased = exp019_graddiff_erased（GradDiff擦除后DiT）+ 原始T5/VAE

评测集：nudity domain 99条（来自 VU benchmark_wan.jsonl）

输出：outputs/exp019/<arm>/<idx:03d>.mp4 + gen_params.json
"""
import argparse
import glob
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# ---------------- 常量 ----------------
VU_ROOT = os.getenv("VU_ROOT", "../video-unlearning")

# 臂定义
ARMS = {
    "graddiff_base": {
        "dit_id": "Wan-AI/Wan2.2-TI2V-5B",
        "t5_vae_id": "Wan-AI/Wan2.2-TI2V-5B",
    },
    "graddiff_erased": {
        "dit_id": "wan5b/exp019_graddiff_erased",
        "t5_vae_id": "Wan-AI/Wan2.2-TI2V-5B",
    },
    "graddiff_finetuned": {
        "dit_id": "finetune/exp019_graddiff_ft",
        "t5_vae_id": "Wan-AI/Wan2.2-TI2V-5B",
    },
    "graddiff_distilled_epoch9": {
        "dit_id": "train/exp019_graddiff_distill",
        "dit_file": "epoch-9.safetensors",
        "t5_vae_id": "Wan-AI/Wan2.2-TI2V-5B",
    },
}

# DiT文件
DIT_FILES = ["diffusion_pytorch_model*.safetensors"]

# T5/VAE文件
T5_VAE_FILES = [
    "models_t5_umt5-xxl-enc-bf16.safetensors",
    "Wan2.2_VAE.safetensors",
]

# 生成参数
DEFAULT_GEN_PARAMS = {
    "num_frames": 17,
    "height": 480,
    "width": 720,
    "num_inference_steps": 50,
    "guidance_scale": 5.0,
    "flow_shift": 5.0,
    "fps": 15,
}

OUTPUT_ROOT = "outputs/exp019"


def load_prompts():
    """加载 nudity 评测 prompts (99条)"""
    prompts_file = "data/exp015/nudity_eval_prompts.jsonl"
    if not os.path.isfile(prompts_file):
        raise FileNotFoundError(f"prompts 文件不存在: {prompts_file}")

    prompts = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data)

    print(f"✓ 加载 {len(prompts)} 条 prompts")
    return prompts


def validate_arm(arm_name):
    """验证臂配置"""
    if arm_name not in ARMS:
        raise ValueError(f"未知臂: {arm_name}，可用: {list(ARMS.keys())}")

    arm = ARMS[arm_name]
    dit_id = arm["dit_id"]
    t5_vae_id = arm["t5_vae_id"]

    # 检查DiT目录和文件
    dit_dir = os.path.join("models", dit_id)
    if not os.path.isdir(dit_dir):
        raise FileNotFoundError(f"DiT目录不存在: {dit_dir}")

    # 如果指定了特定的 dit_file（如蒸馏checkpoint），检查该文件
    if "dit_file" in arm:
        dit_file = os.path.join(dit_dir, arm["dit_file"])
        if not os.path.isfile(dit_file):
            raise FileNotFoundError(f"缺少DiT文件: {dit_file}")
        print(f"  使用DiT checkpoint: {arm['dit_file']}")
    else:
        # 否则检查标准DiT文件
        for pattern in DIT_FILES:
            files = glob.glob(os.path.join(dit_dir, pattern))
            if not files:
                raise FileNotFoundError(f"缺少DiT文件: {dit_dir}/{pattern}")

    # 检查T5/VAE目录和文件
    t5_vae_dir = os.path.join("models", t5_vae_id)
    if not os.path.isdir(t5_vae_dir):
        raise FileNotFoundError(f"T5/VAE目录不存在: {t5_vae_dir}")

    for pattern in T5_VAE_FILES:
        files = glob.glob(os.path.join(t5_vae_dir, pattern))
        if not files:
            raise FileNotFoundError(f"缺少T5/VAE文件: {t5_vae_dir}/{pattern}")

    print(f"✓ 臂 {arm_name} 验证通过")
    print(f"  DiT: {dit_dir}")
    print(f"  T5/VAE: {t5_vae_dir}")
    return arm


def build_parser():
    parser = argparse.ArgumentParser(description="Exp019 视频生成")
    parser.add_argument("--arm", type=str, required=True, choices=list(ARMS.keys()),
                        help="生成臂")
    parser.add_argument("--start", type=int, default=0, help="起始索引（含）")
    parser.add_argument("--end", type=int, default=99, help="结束索引（不含）")
    parser.add_argument("--frames", type=int, default=DEFAULT_GEN_PARAMS["num_frames"],
                        help="视频帧数")
    parser.add_argument("--height", type=int, default=DEFAULT_GEN_PARAMS["height"],
                        help="视频高度")
    parser.add_argument("--width", type=int, default=DEFAULT_GEN_PARAMS["width"],
                        help="视频宽度")
    parser.add_argument("--steps", type=int, default=DEFAULT_GEN_PARAMS["num_inference_steps"],
                        help="推理步数")
    parser.add_argument("--cfg", type=float, default=DEFAULT_GEN_PARAMS["guidance_scale"],
                        help="CFG scale")
    parser.add_argument("--flow-shift", type=float, default=DEFAULT_GEN_PARAMS["flow_shift"],
                        help="Flow shift")
    parser.add_argument("--fps", type=int, default=DEFAULT_GEN_PARAMS["fps"],
                        help="FPS")
    parser.add_argument("--dry-run", action="store_true", help="预览模式（不加载模型）")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 验证环境
    if not args.dry_run:
        if os.getenv("DIFFSYNTH_SKIP_DOWNLOAD") != "true":
            print("警告: 建议设置 DIFFSYNTH_SKIP_DOWNLOAD=true 禁止在线下载")

    # 加载 prompts
    prompts = load_prompts()
    if args.end > len(prompts):
        args.end = len(prompts)

    # 验证臂
    arm = validate_arm(args.arm)

    # 输出目录
    output_dir = os.path.join(OUTPUT_ROOT, args.arm)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Exp019 视频生成")
    print(f"  臂: {args.arm}")
    print(f"  范围: [{args.start}, {args.end}) / {len(prompts)}")
    print(f"  输出: {output_dir}")
    print(f"  参数: {args.frames}f {args.height}x{args.width} steps={args.steps} cfg={args.cfg}")
    print(f"{'='*80}\n")

    if args.dry_run:
        print("[DRY-RUN] 预览模式，不实际生成")
        for idx in range(args.start, args.end):
            prompt_data = prompts[idx]
            output_path = os.path.join(output_dir, f"{idx:03d}.mp4")
            print(f"[{idx:03d}] {prompt_data['prompt'][:80]}...")
            print(f"       seed={prompt_data['seed']} -> {output_path}")
        print(f"\n共 {args.end - args.start} 个视频")
        return

    # 加载模型
    print("加载模型...")
    import torch
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.utils.data import save_video

    # 构建模型配置
    model_configs = []

    # DiT配置
    if "dit_file" in arm:
        # 蒸馏/微调checkpoint：使用特定文件
        model_configs.append(
            ModelConfig(model_id=arm['dit_id'], origin_file_pattern=arm['dit_file'])
        )
    else:
        # 标准模型：使用通配符
        for pattern in DIT_FILES:
            model_configs.append(
                ModelConfig(model_id=arm['dit_id'], origin_file_pattern=pattern)
            )

    # T5/VAE配置
    for pattern in T5_VAE_FILES:
        model_configs.append(
            ModelConfig(model_id=arm['t5_vae_id'], origin_file_pattern=pattern)
        )

    # Tokenizer 配置（使用 1.3B 的 tokenizer）
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

    print("✓ 模型加载完成\n")

    # 生成参数
    gen_params = {
        "num_frames": args.frames,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "cfg_scale": args.cfg,
        "sigma_shift": args.flow_shift,  # CLI用flow_shift，API用sigma_shift
    }

    # 保存生成参数
    params_path = os.path.join(output_dir, "gen_params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({
            "arm": args.arm,
            "dit_id": arm["dit_id"],
            "t5_vae_id": arm["t5_vae_id"],
            "generation_params": gen_params,
            "num_prompts": len(prompts),
            "range": [args.start, args.end],
        }, f, indent=2, ensure_ascii=False)

    # 生成视频
    start_time = time.time()
    for idx in range(args.start, args.end):
        prompt_data = prompts[idx]
        prompt = prompt_data["prompt"]
        seed = prompt_data["seed"]
        output_path = os.path.join(output_dir, f"{idx:03d}.mp4")

        if os.path.exists(output_path):
            print(f"[{idx:03d}] 已存在，跳过")
            continue

        print(f"[{idx:03d}] {prompt[:60]}...")

        t0 = time.time()
        video = pipe(
            prompt=prompt,
            seed=seed,
            **gen_params,
        )
        t_gen = time.time() - t0

        # 保存视频
        save_video(video, output_path, fps=args.fps)

        elapsed = time.time() - start_time
        avg_time = elapsed / (idx - args.start + 1)
        remaining = avg_time * (args.end - idx - 1)

        print(f"       ✓ {t_gen:.1f}s -> {output_path}")
        print(f"       进度: {idx - args.start + 1}/{args.end - args.start} "
              f"已用 {elapsed/60:.1f}m 预计剩余 {remaining/60:.1f}m")

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"✓ 生成完成")
    print(f"  生成数量: {args.end - args.start}")
    print(f"  总耗时: {total_time/60:.1f} 分钟")
    print(f"  平均: {total_time/(args.end - args.start):.1f} 秒/视频")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
