#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp015 8臂视频生成（4方法 × 2版本：erased + base）

8臂定义：
    esd_erased         = exp015_esd_erased         + LoRA epoch-1
    esd_base          = Wan2.2-TI2V-5B             + LoRA epoch-1
    npo_erased        = exp015_npo_erased          + LoRA epoch-1
    npo_base          = Wan2.2-TI2V-5B             + LoRA epoch-1
    grad_ascent_erased = exp015_grad_ascent_erased + LoRA epoch-1
    grad_ascent_base   = Wan2.2-TI2V-5B             + LoRA epoch-1
    anchor_distill_erased = exp015_anchor_distill_erased + LoRA epoch-1
    anchor_distill_base   = Wan2.2-TI2V-5B             + LoRA epoch-1

评测集：nudity domain 99条（来自 VU benchmark_wan.jsonl）

输出：outputs/exp015/<arm>/<idx:03d>.mp4 + gen_params.json

用法：
    # 预览
    python3 scripts/exp015_generate_videos.py --arm esd_erased --dry-run

    # 生成（集群）
    python -u scripts/exp015_generate_videos.py --arm esd_erased --start 0 --end 99
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

# 8臂定义
ARMS = {
    "esd_erased": {
        "model_id": "wan5b/exp015_esd_erased",
        "lora": "models/finetune/exp015_esd_erased_lora10e2/epoch-1.safetensors",
    },
    "esd_base": {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "lora": "models/finetune/exp015_esd_erased_lora10e2/epoch-1.safetensors",
    },
    "npo_erased": {
        "model_id": "wan5b/exp015_npo_erased",
        "lora": "models/finetune/exp015_npo_erased_lora10e2/epoch-1.safetensors",
    },
    "npo_base": {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "lora": "models/finetune/exp015_npo_erased_lora10e2/epoch-1.safetensors",
    },
    "grad_ascent_erased": {
        "model_id": "wan5b/exp015_grad_ascent_erased",
        "lora": "models/finetune/exp015_grad_ascent_erased_lora10e2/epoch-1.safetensors",
    },
    "grad_ascent_base": {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "lora": "models/finetune/exp015_grad_ascent_erased_lora10e2/epoch-1.safetensors",
    },
    "anchor_distill_erased": {
        "model_id": "wan5b/exp015_anchor_distill_erased",
        "lora": "models/finetune/exp015_anchor_distill_erased_lora10e2/epoch-1.safetensors",
    },
    "anchor_distill_base": {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "lora": "models/finetune/exp015_anchor_distill_erased_lora10e2/epoch-1.safetensors",
    },
}

# 模型文件要求
MODEL_FILES = [
    "diffusion_pytorch_model*.safetensors",
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

OUTPUT_ROOT = "outputs/exp015"


def load_prompts():
    """加载 nudity 评测 prompts (99条)"""
    prompts_file = "data/exp015/nudity_eval_prompts.jsonl"
    if not os.path.isfile(prompts_file):
        raise FileNotFoundError(f"评测 prompts 文件不存在: {prompts_file}")

    prompts = []
    with open(prompts_file) as f:
        for line in f:
            data = json.loads(line)
            prompts.append({
                "prompt": data["prompt"],
                "seed": data["seed"],
            })

    print(f"✓ 加载 {len(prompts)} 条评测 prompts")
    return prompts


def validate_arm(arm_name):
    """验证臂配置和文件存在性"""
    if arm_name not in ARMS:
        raise ValueError(f"未知臂: {arm_name}，可用: {list(ARMS.keys())}")

    arm = ARMS[arm_name]
    model_id = arm["model_id"]
    lora_path = arm["lora"]

    # 检查模型目录
    model_dir = os.path.join("models", model_id)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    # 检查模型文件
    for pattern in MODEL_FILES:
        files = glob.glob(os.path.join(model_dir, pattern))
        if not files:
            raise FileNotFoundError(f"缺少文件: {model_dir}/{pattern}")

    # 检查 LoRA
    if lora_path and not os.path.isfile(lora_path):
        raise FileNotFoundError(f"LoRA 文件不存在: {lora_path}")

    print(f"✓ 臂 {arm_name} 验证通过")
    print(f"  模型: {model_dir}")
    print(f"  LoRA: {lora_path}")
    return arm


def build_parser():
    parser = argparse.ArgumentParser(description="Exp015 8臂视频生成")
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
    print(f"Exp015 视频生成")
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
    model_configs = [
        ModelConfig(model_id=arm['model_id'], origin_file_pattern=pattern)
        for pattern in MODEL_FILES
    ]

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

    # 加载 LoRA
    if arm["lora"]:
        print(f"加载 LoRA: {arm['lora']}")
        pipe.load_lora(pipe.dit, arm["lora"], alpha=1.0)

    print("✓ 模型加载完成\n")

    # 生成视频
    for idx in range(args.start, args.end):
        prompt_data = prompts[idx]
        prompt = prompt_data["prompt"]
        seed = prompt_data["seed"]
        output_path = os.path.join(output_dir, f"{idx:03d}.mp4")

        if os.path.exists(output_path):
            print(f"[{idx:03d}] 跳过（已存在）: {output_path}")
            continue

        print(f"[{idx:03d}] 生成中...")
        print(f"  prompt: {prompt[:100]}...")
        print(f"  seed: {seed}")

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

        print(f"  ✓ 已保存: {output_path} ({elapsed:.1f}s)\n")

    # 保存生成参数
    params_file = os.path.join(OUTPUT_ROOT, "gen_params.json")
    if not os.path.exists(params_file):
        with open(params_file, "w") as f:
            json.dump({
                "num_frames": args.frames,
                "height": args.height,
                "width": args.width,
                "num_inference_steps": args.steps,
                "guidance_scale": args.cfg,
                "flow_shift": args.flow_shift,
                "fps": args.fps,
            }, f, indent=2)
        print(f"✓ 生成参数已保存: {params_file}")

    print(f"\n完成！生成 {args.end - args.start} 个视频")


if __name__ == "__main__":
    main()
