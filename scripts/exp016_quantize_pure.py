#!/usr/bin/env python3
"""
Exp016 Stage 1: 量化5个模型为NF4（纯模型，无LoRA）

量化目标:
  - 5个模型: baseline, esd_erased, npo_erased, grad_ascent_erased, anchor_distill_erased
  - 每个模型都是纯模型（无LoRA）

量化方法: NF4 (bitsandbytes)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from safetensors.torch import save_file

# 5个模型配置
MODELS = {
    "baseline": {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "description": "Original Wan2.2-TI2V-5B",
    },
    "esd_erased": {
        "model_id": "wan5b/exp015_esd_erased",
        "description": "ESD erased (pure)",
    },
    "npo_erased": {
        "model_id": "wan5b/exp015_npo_erased",
        "description": "NPO erased (pure)",
    },
    "grad_ascent_erased": {
        "model_id": "wan5b/exp015_grad_ascent_erased",
        "description": "GradAscent erased (pure)",
    },
    "anchor_distill_erased": {
        "model_id": "wan5b/exp015_anchor_distill_erased",
        "description": "AnchorDistill erased (pure)",
    },
}


def quantize_model(model_name: str, output_dir: Path):
    """
    量化单个模型

    策略：
    1. 加载模型（纯模型，无LoRA）
    2. 量化 DiT（NF4）
    3. 保存量化后的 DiT 权重
    """
    if model_name not in MODELS:
        raise ValueError(f"未知的模型: {model_name}")

    model_config = MODELS[model_name]
    model_id = model_config["model_id"]

    print(f"\n{'='*60}")
    print(f"量化: {model_name}")
    print(f"{'='*60}")
    print(f"模型ID: {model_id}")
    print(f"描述: {model_config['description']}")
    print(f"输出目录: {output_dir}")
    print("")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用 diffsynth 的量化流程
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.core.quant import QuantizeConfig

    # 模型文件
    MODEL_FILES = [
        "diffusion_pytorch_model*.safetensors",
        "models_t5_umt5-xxl-enc-bf16.safetensors",
        "Wan2.2_VAE.safetensors",
    ]

    print("步骤1: 加载原始模型...")
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

    print("✓ 模型加载完成")

    # 量化 DiT
    print("\n步骤2: 应用量化配置（NF4）...")
    quant_config = QuantizeConfig(method="bitsandbytes_nf4")
    quant_config.quantize_model(pipe.dit, compute_device="cuda", model_device="cuda")
    print("✓ 量化完成")

    # 保存量化后的 DiT
    print("\n步骤3: 保存量化权重...")
    quantized_dit_path = output_dir / "diffusion_pytorch_model.safetensors"

    state_dict = pipe.dit.state_dict()
    save_file(state_dict, str(quantized_dit_path))

    print(f"✓ 保存到: {quantized_dit_path}")

    # 保存元信息
    meta_file = output_dir / "quantization_meta.json"
    with open(meta_file, "w") as f:
        json.dump({
            "model_name": model_name,
            "model_id": model_id,
            "description": model_config["description"],
            "quantization_method": "bitsandbytes_nf4",
            "quantized_component": "dit",
        }, f, indent=2)

    print(f"✓ 元信息保存到: {meta_file}")
    print(f"\n{'='*60}")
    print(f"✅ {model_name} 量化完成")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Exp016 Stage 1: 模型量化")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()),
                        help="模型名称")
    parser.add_argument("--output-dir", type=str,
                        default=None,
                        help="输出目录（默认: models/quantized/exp016_<model>_nf4）")

    args = parser.parse_args()

    # 确定输出目录
    if args.output_dir is None:
        args.output_dir = f"models/quantized/exp016_{args.model}_nf4"

    output_dir = Path(args.output_dir)

    print(f"\nExp016 Stage 1: 量化 {args.model}")
    print(f"输出: {output_dir}\n")

    quantize_model(args.model, output_dir)


if __name__ == "__main__":
    main()
