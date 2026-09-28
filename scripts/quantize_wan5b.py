"""
Wan2.2-TI2V-5B 量化脚本

基于 diffsynth.core.quant 量化框架，支持 bitsandbytes (nf4/fp4) 量化。
适配自 quantize.py，专门针对 Wan2.2-TI2V-5B (5B参数) 模型。

三种模式:
  1. infer (默认): 在线量化推理 — 加载 fp 模型并即时量化，直接生成视频验证效果
  2. save:         保存量化 checkpoint — 将量化后的权重保存到磁盘
  3. load:         加载已保存的量化 checkpoint 并推理

用法示例:
    # 在线量化推理（默认，最简单）
    python scripts/quantize_wan5b.py --arm base --method bitsandbytes_nf4

    # 保存量化 checkpoint
    python scripts/quantize_wan5b.py --mode save --arm base --method bitsandbytes_nf4 \
        --output_path ./models/quantized/Wan2.2-TI2V-5B_base_nf4

    # 加载已保存的量化 checkpoint 并推理
    python scripts/quantize_wan5b.py --mode load --arm base --method bitsandbytes_nf4 \
        --quantized_path ./models/quantized/Wan2.2-TI2V-5B_base_nf4.safetensors

支持的arm:
    base         - Wan2.2-TI2V-5B 基座模型（未擦除）
    erased       - Wan2.2-TI2V-5B 擦除版本
    base_ft      - base + LoRA微调
    erased_ft    - erased + LoRA微调

可用量化方法:
    bitsandbytes_nf4   — 4bit NF4，仅权重量化
    bitsandbytes_fp4   — 4bit FP4，仅权重量化

依赖:
    bitsandbytes:  pip install bitsandbytes
"""
import argparse
import json
import os
import sys

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffsynth.core.quant import QuantizeConfig, describe_quant_method

# Wan2.2-TI2V-5B 四臂配置（与 generate_wan5b_eval.py 保持一致）
ARMS = {
    "base":      {"model_id": "Wan-AI/Wan2.2-TI2V-5B",          "lora": None},
    "erased":    {"model_id": "Wan-AI/Wan2.2-TI2V-5B-erased",   "lora": None},
    "base_ft":   {"model_id": "Wan-AI/Wan2.2-TI2V-5B",          "lora": "models/train/Wan2.2-TI2V-5B_base_lora10e2/epoch-0.safetensors"},
    "erased_ft": {"model_id": "Wan-AI/Wan2.2-TI2V-5B-erased",   "lora": "models/train/Wan2.2-TI2V-5B_erased_lora10e2/epoch-0.safetensors"},
}


def build_pipeline(arm, method, device="cuda", quantized_dit_path=None):
    """
    构建 Wan2.2-TI2V-5B 量化 pipeline

    Args:
        arm: 模型臂 (base/erased/base_ft/erased_ft)
        method: 量化方法
        device: 设备
        quantized_dit_path: 已保存的量化 DiT 路径 (mode=load时使用)
    """
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

    if arm not in ARMS:
        raise ValueError(f"未知的arm: {arm}，可选: {list(ARMS.keys())}")

    arm_config = ARMS[arm]
    model_id = arm_config["model_id"]
    lora_path = arm_config["lora"]

    print(f"\n{'='*70}")
    print(f"构建 Wan2.2-TI2V-5B Pipeline")
    print(f"  Arm: {arm}")
    print(f"  Model ID: {model_id}")
    print(f"  LoRA: {lora_path if lora_path else 'None'}")
    print(f"  Quantization: {method}")
    print(f"{'='*70}\n")

    # 量化配置
    quant_config = QuantizeConfig(method=method) if method != "none" else None

    # 模型配置
    model_configs = []

    # DiT 配置
    if quantized_dit_path:
        # 加载已保存的量化 checkpoint
        print(f"加载量化 DiT: {quantized_dit_path}")
        model_configs.append(ModelConfig(path=quantized_dit_path))
    else:
        # 对于带 LoRA 的模型：先加载base模型（不量化），加载LoRA，再量化
        # 对于不带 LoRA 的模型：直接在线量化
        if lora_path:
            print(f"检测到 LoRA，将先加载base模型，然后加载LoRA，最后量化")
            model_configs.append(ModelConfig(
                model_id=model_id,
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                # 不在 ModelConfig 中量化
            ))
        else:
            # 无 LoRA，直接在线量化
            model_configs.append(ModelConfig(
                model_id=model_id,
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                quantize=quant_config,
            ))

    # T5 编码器和 VAE（共享，使用 safetensors 格式）
    shared_dir = "models/DiffSynth-Studio/Wan-Series-Converted-Safetensors"
    model_configs.extend([
        ModelConfig(path=f"{shared_dir}/models_t5_umt5-xxl-enc-bf16.safetensors"),
        ModelConfig(path=f"{shared_dir}/Wan2.2_VAE.safetensors"),
    ])

    # 构建 pipeline
    # Wan2.2-TI2V-5B 没有独立 tokenizer，需要使用 Wan2.1-T2V-1.3B 的 tokenizer
    tokenizer_model_id = "Wan-AI/Wan2.1-T2V-1.3B"
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=ModelConfig(model_id=tokenizer_model_id, origin_file_pattern="google/umt5-xxl/"),
    )

    # 加载 LoRA（如果需要）
    if lora_path:
        print(f"加载 LoRA: {lora_path} (alpha=1)")
        pipe.load_lora(pipe.dit, lora_path, alpha=1)

        # LoRA 加载后，再量化
        if quant_config and not quantized_dit_path:
            print(f"LoRA已加载，现在量化模型...")
            quant_config.quantize_model(pipe.dit, compute_device=device, model_device=device)

    return pipe


def run_inference(pipe, args):
    """运行推理并保存视频"""
    from diffsynth.utils.data import save_video

    print(f"\n推理中 (steps={args.num_inference_steps}, cfg={args.cfg_scale})...")
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
    )

    output_path = args.output_video
    save_video(video, output_path, fps=args.fps, quality=5)
    print(f"✅ 视频已保存到: {output_path}")

    # 输出量化统计
    if hasattr(pipe.dit, "quant_state_dict"):
        print("\n量化统计:")
        quant_info = pipe.dit.quant_state_dict()
        if quant_info:
            print(json.dumps(quant_info, indent=2, ensure_ascii=False))


def mode_infer(args):
    """模式1: 在线量化推理"""
    print(f"\n[模式] 在线量化推理 (method={args.method})")
    pipe = build_pipeline(args.arm, args.method, args.device)
    run_inference(pipe, args)


def mode_save(args):
    """模式2: 保存量化 checkpoint"""
    print(f"\n[模式] 保存量化 checkpoint")

    # 构建量化 pipeline
    pipe = build_pipeline(args.arm, args.method, args.device)

    # 保存量化后的 DiT
    output_path = args.output_path
    if not output_path.endswith(".safetensors"):
        output_path = output_path + ".safetensors"

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    print(f"量化 DiT (方法: {args.method})...")
    quant_config = QuantizeConfig.from_method_name(args.method)
    quant_config.quantize_model(pipe.dit, compute_device="cuda", model_device="cpu")

    print(f"保存量化权重到: {output_path}")
    from safetensors.torch import save_file
    save_file(pipe.dit.state_dict(), output_path)

    print(f"✅ 量化 checkpoint 已保存")

    # 保存元信息
    meta_path = output_path.replace(".safetensors", "_meta.json")
    meta = {
        "arm": args.arm,
        "model_id": ARMS[args.arm]["model_id"],
        "lora": ARMS[args.arm]["lora"],
        "quantization_method": args.method,
        "original_dtype": "bfloat16",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"✅ 元信息已保存: {meta_path}")


def mode_load(args):
    """模式3: 加载量化 checkpoint 并推理"""
    print(f"\n[模式] 加载量化 checkpoint")

    if not os.path.exists(args.quantized_path):
        raise FileNotFoundError(f"量化 checkpoint 不存在: {args.quantized_path}")

    # 构建 pipeline（使用量化的 DiT）
    pipe = build_pipeline(args.arm, "none", args.device, quantized_dit_path=args.quantized_path)
    run_inference(pipe, args)


def main():
    parser = argparse.ArgumentParser(description="Wan2.2-TI2V-5B 量化工具")

    # 模式
    parser.add_argument("--mode", choices=["infer", "save", "load"], default="infer",
                        help="运行模式: infer(在线量化推理), save(保存量化checkpoint), load(加载量化checkpoint推理)")

    # 模型配置
    parser.add_argument("--arm", type=str, required=True, choices=list(ARMS.keys()),
                        help="模型臂: base, erased, base_ft, erased_ft")
    parser.add_argument("--method", type=str, default="bitsandbytes_nf4",
                        help="量化方法 (infer/save模式)")
    parser.add_argument("--device", type=str, default="cuda", help="设备")

    # 保存/加载路径
    parser.add_argument("--output_path", type=str, default=None,
                        help="保存量化checkpoint的路径 (save模式)")
    parser.add_argument("--quantized_path", type=str, default=None,
                        help="已保存的量化checkpoint路径 (load模式)")

    # 推理参数
    parser.add_argument("--prompt", type=str,
                        default="A woman in red dress dancing in the rain, cinematic lighting",
                        help="文本提示词")
    parser.add_argument("--negative_prompt", type=str, default="", help="负面提示词")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--height", type=int, default=480, help="视频高度")
    parser.add_argument("--width", type=int, default=720, help="视频宽度")
    parser.add_argument("--num_frames", type=int, default=17, help="视频帧数")
    parser.add_argument("--fps", type=int, default=15, help="FPS")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output_video", type=str, default="output_quantized.mp4",
                        help="输出视频路径 (infer/load模式)")

    # 工具命令
    parser.add_argument("--list_methods", action="store_true", help="列出可用的量化方法")

    args = parser.parse_args()

    if args.list_methods:
        print("\n可用的量化方法:")
        for method in ["bitsandbytes_nf4", "bitsandbytes_fp4"]:
            print(f"\n{method}:")
            print(describe_quant_method(method))
        return

    # 验证参数
    if args.mode == "save" and not args.output_path:
        parser.error("save模式需要 --output_path")
    if args.mode == "load" and not args.quantized_path:
        parser.error("load模式需要 --quantized_path")

    # 运行对应模式
    if args.mode == "infer":
        mode_infer(args)
    elif args.mode == "save":
        mode_save(args)
    elif args.mode == "load":
        mode_load(args)


if __name__ == "__main__":
    main()
