#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp017 蒸馏模型批量视频生成

基于exp015脚本改编，用于蒸馏模型（4步推理）的批量生成

5个蒸馏模型：
    base_distill          = exp017_base_distill epoch-9
    grad_ascent_distill   = exp017_grad_ascent_distill epoch-9
    esd_distill           = exp017_esd_distill epoch-9
    npo_distill           = exp017_npo_distill epoch-9
    anchor_distill_distill = exp017_anchor_distill_distill epoch-9

评测集：nudity domain 99条（来自 VU benchmark_wan.jsonl）

输出：outputs/exp017/<arm>/<idx:03d>.mp4 + gen_params.json

用法：
    # 预览
    python scripts/exp017_generate_videos.py --arm base_distill --dry-run

    # 生成（集群）
    python -u scripts/exp017_generate_videos.py --arm base_distill --start 0 --end 99
"""
import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# ---------------- 常量 ----------------
VU_ROOT = os.getenv("VU_ROOT", "../video-unlearning")

# 5个蒸馏模型定义
DISTILL_MODELS = {
    "base_distill": {
        "distill_path": "models/train/exp017_base_distill/epoch-9.safetensors",
    },
    "grad_ascent_distill": {
        "distill_path": "models/train/exp017_grad_ascent_distill/epoch-9.safetensors",
    },
    "esd_distill": {
        "distill_path": "models/train/exp017_esd_distill/epoch-9.safetensors",
    },
    "npo_distill": {
        "distill_path": "models/train/exp017_npo_distill/epoch-9.safetensors",
    },
    "anchor_distill_distill": {
        "distill_path": "models/train/exp017_anchor_distill_distill/epoch-9.safetensors",
    },
}

# T5和VAE（共享）
SHARED_MODELS = [
    "models_t5_umt5-xxl-enc-bf16.safetensors",
    "Wan2.2_VAE.safetensors",
]

# 生成参数（蒸馏模型专用）
DEFAULT_GEN_PARAMS = {
    "num_frames": 17,
    "height": 480,
    "width": 736,
    "num_inference_steps": 4,  # 蒸馏目标步数
    "cfg_scale": 1.0,           # 蒸馏通常用1.0
    "sigma_shift": 7.0,
    "fps": 8,
}

OUTPUT_ROOT = "outputs/exp017"

# ---------------- 辅助函数 ----------------
def load_prompts():
    """加载 VU benchmark nudity 99条"""
    jsonl_path = os.path.join(VU_ROOT, "data", "benchmark_wan.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"找不到 benchmark: {jsonl_path}")

    prompts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get("domain") == "nudity":
                prompts.append({
                    "prompt": data["prompt"],
                    "seed": data.get("seed", 42),
                })

    if len(prompts) != 99:
        print(f"警告: nudity 条目数 = {len(prompts)}, 预期 99")

    return prompts


def validate_arm(arm_name):
    """验证臂名称并返回配置"""
    if arm_name not in DISTILL_MODELS:
        raise ValueError(f"未知的臂: {arm_name}\n可选: {list(DISTILL_MODELS.keys())}")

    arm = DISTILL_MODELS[arm_name]

    # 检查蒸馏模型文件
    if not os.path.exists(arm["distill_path"]):
        raise FileNotFoundError(f"蒸馏模型文件不存在: {arm['distill_path']}")

    return arm


def build_parser():
    parser = argparse.ArgumentParser(description="Exp017 蒸馏模型批量视频生成")
    parser.add_argument("--arm", required=True, help="模型臂名称")
    parser.add_argument("--start", type=int, default=0, help="起始索引（含）")
    parser.add_argument("--end", type=int, default=99, help="结束索引（不含）")
    parser.add_argument("--frames", type=int, default=DEFAULT_GEN_PARAMS["num_frames"],
                        help="视频帧数")
    parser.add_argument("--height", type=int, default=DEFAULT_GEN_PARAMS["height"],
                        help="视频高度")
    parser.add_argument("--width", type=int, default=DEFAULT_GEN_PARAMS["width"],
                        help="视频宽度")
    parser.add_argument("--steps", type=int, default=DEFAULT_GEN_PARAMS["num_inference_steps"],
                        help="推理步数（蒸馏模型默认4步）")
    parser.add_argument("--cfg", type=float, default=DEFAULT_GEN_PARAMS["cfg_scale"],
                        help="CFG scale（蒸馏通常1.0）")
    parser.add_argument("--sigma-shift", type=float, default=DEFAULT_GEN_PARAMS["sigma_shift"],
                        help="Sigma shift")
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
    print(f"Exp017 蒸馏模型视频生成")
    print(f"  臂: {args.arm}")
    print(f"  蒸馏模型: {arm['distill_path']}")
    print(f"  推理步数: {args.steps}步（蒸馏）")
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
    print("加载蒸馏模型...")
    import torch
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.utils.data import save_video

    # 构建模型配置
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=arm["distill_path"]),  # 蒸馏后的DiT
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.safetensors"),
        ],
        tokenizer_config=ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="google/umt5-xxl/"
        ),
    )

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
        print(f"  提示词: {prompt[:100]}...")
        print(f"  种子: {seed}")

        try:
            start_time = time.time()

            video = pipe(
                prompt=prompt,
                num_inference_steps=args.steps,
                cfg_scale=args.cfg,
                sigma_shift=args.sigma_shift,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                seed=seed,
            )

            # 保存视频
            save_video(video, output_path, fps=args.fps, quality=5)

            elapsed = time.time() - start_time
            print(f"  ✓ 完成 ({elapsed:.1f}秒) -> {output_path}")

            # 保存生成参数
            params_path = output_path.replace(".mp4", ".json")
            with open(params_path, "w") as f:
                json.dump({
                    "prompt": prompt,
                    "seed": seed,
                    "num_inference_steps": args.steps,
                    "cfg_scale": args.cfg,
                    "sigma_shift": args.sigma_shift,
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.frames,
                    "fps": args.fps,
                    "distill_model": arm["distill_path"],
                    "elapsed_seconds": elapsed,
                }, f, indent=2)

        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("生成完成！")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
