#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp019 Phase 6: 微调后视频生成

生成臂：graddiff_finetuned_epoch{1..20}（擦除后模型 + 微调LoRA）
"""
import argparse
import glob
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# 配置
ERASED_MODEL_ID = "wan5b/exp019_graddiff_erased"
LORA_BASE_PATH = "models/finetune/exp019_graddiff_erased_lora"
T5_VAE_ID = "Wan-AI/Wan2.2-TI2V-5B"

DIT_FILES = ["diffusion_pytorch_model*.safetensors"]
T5_VAE_FILES = [
    "models_t5_umt5-xxl-enc-bf16.safetensors",
    "Wan2.2_VAE.safetensors",
]

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


def validate_config(epoch):
    """验证配置"""
    # 检查DiT
    dit_dir = os.path.join("models", ERASED_MODEL_ID)
    if not os.path.isdir(dit_dir):
        raise FileNotFoundError(f"DiT目录不存在: {dit_dir}")

    for pattern in DIT_FILES:
        files = glob.glob(os.path.join(dit_dir, pattern))
        if not files:
            raise FileNotFoundError(f"缺少DiT文件: {dit_dir}/{pattern}")

    # 检查T5/VAE
    t5_vae_dir = os.path.join("models", T5_VAE_ID)
    if not os.path.isdir(t5_vae_dir):
        raise FileNotFoundError(f"T5/VAE目录不存在: {t5_vae_dir}")

    for pattern in T5_VAE_FILES:
        files = glob.glob(os.path.join(t5_vae_dir, pattern))
        if not files:
            raise FileNotFoundError(f"缺少T5/VAE文件: {t5_vae_dir}/{pattern}")

    # 检查LoRA
    lora_path = os.path.join(LORA_BASE_PATH, f"epoch-{epoch}.safetensors")
    if not os.path.isfile(lora_path):
        raise FileNotFoundError(f"LoRA checkpoint不存在: {lora_path}")

    print(f"✓ 配置验证通过")
    print(f"  DiT: {dit_dir}")
    print(f"  T5/VAE: {t5_vae_dir}")
    print(f"  LoRA: {lora_path}")

    return lora_path


def build_parser():
    parser = argparse.ArgumentParser(description="Exp019 微调后视频生成")
    parser.add_argument("--epoch", type=int, required=True, help="微调epoch (1-20)")
    parser.add_argument("--start", type=int, default=0, help="起始索引")
    parser.add_argument("--end", type=int, default=99, help="结束索引")
    parser.add_argument("--frames", type=int, default=DEFAULT_GEN_PARAMS["num_frames"])
    parser.add_argument("--height", type=int, default=DEFAULT_GEN_PARAMS["height"])
    parser.add_argument("--width", type=int, default=DEFAULT_GEN_PARAMS["width"])
    parser.add_argument("--steps", type=int, default=DEFAULT_GEN_PARAMS["num_inference_steps"])
    parser.add_argument("--cfg", type=float, default=DEFAULT_GEN_PARAMS["guidance_scale"])
    parser.add_argument("--flow-shift", type=float, default=DEFAULT_GEN_PARAMS["flow_shift"])
    parser.add_argument("--fps", type=int, default=DEFAULT_GEN_PARAMS["fps"])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.dry_run and os.getenv("DIFFSYNTH_SKIP_DOWNLOAD") != "true":
        print("警告: 建议设置 DIFFSYNTH_SKIP_DOWNLOAD=true")

    # 加载prompts
    prompts = load_prompts()
    if args.end > len(prompts):
        args.end = len(prompts)

    # 验证配置
    lora_path = validate_config(args.epoch)

    # 输出目录
    arm_name = f"graddiff_finetuned_epoch{args.epoch}"
    output_dir = os.path.join(OUTPUT_ROOT, arm_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Exp019 Phase 6 - 微调后生成")
    print(f"  臂: {arm_name}")
    print(f"  范围: [{args.start}, {args.end}) / {len(prompts)}")
    print(f"  输出: {output_dir}")
    print(f"{'='*80}\n")

    if args.dry_run:
        print("[DRY-RUN] 预览模式")
        for idx in range(args.start, args.end):
            print(f"[{idx:03d}] {prompts[idx]['prompt'][:60]}...")
        return

    # 加载模型
    print("加载模型...")
    import torch
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.utils.data import save_video

    model_configs = []
    for pattern in DIT_FILES:
        model_configs.append(ModelConfig(model_id=ERASED_MODEL_ID, origin_file_pattern=pattern))
    for pattern in T5_VAE_FILES:
        model_configs.append(ModelConfig(model_id=T5_VAE_ID, origin_file_pattern=pattern))

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

    # 加载LoRA
    print(f"加载LoRA: {lora_path}")
    pipe.load_lora(pipe.dit, lora_path, alpha=1.0)
    print("✓ 模型加载完成\n")

    # 生成参数
    gen_params = {
        "num_frames": args.frames,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "cfg_scale": args.cfg,
        "sigma_shift": args.flow_shift,
    }

    # 保存参数
    params_path = os.path.join(output_dir, "gen_params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({
            "arm": arm_name,
            "epoch": args.epoch,
            "dit_id": ERASED_MODEL_ID,
            "lora_path": lora_path,
            "generation_params": gen_params,
        }, f, indent=2, ensure_ascii=False)

    # 生成视频
    start_time = time.time()
    for idx in range(args.start, args.end):
        prompt_data = prompts[idx]
        output_path = os.path.join(output_dir, f"{idx:03d}.mp4")

        if os.path.exists(output_path):
            print(f"[{idx:03d}] 已存在，跳过")
            continue

        print(f"[{idx:03d}] {prompt_data['prompt'][:60]}...")

        t0 = time.time()
        video = pipe(
            prompt=prompt_data["prompt"],
            seed=prompt_data["seed"],
            **gen_params,
        )
        t_gen = time.time() - t0

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
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
