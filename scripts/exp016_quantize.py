#!/usr/bin/env python3
"""
Exp016: 量化 Exp015 的 8 个模型（4方法 × 2版本）

量化目标:
  - 8个模型: esd_erased, esd_base, npo_erased, npo_base,
             grad_ascent_erased, grad_ascent_base,
             anchor_distill_erased, anchor_distill_base
  - 每个模型都是：擦除后模型 + 微调LoRA（或 base + 微调LoRA）

量化方法: NF4 (bitsandbytes)
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from safetensors.torch import save_file

# 8臂配置（与 exp015_generate_videos.py 一致）
ARMS = {
    "esd_erased": {
        "base_model": "wan5b/exp015_esd_erased",
        "lora": "models/finetune/exp015_esd_erased_lora10e2/epoch-1.safetensors",
    },
    "esd_base": {
        "base_model": "Wan-AI/Wan2.2-TI2V-5B",
        "lora": "models/finetune/exp015_esd_erased_lora10e2/epoch-1.safetensors",
    },
    "npo_erased": {
        "base_model": "wan5b/exp015_npo_erased",
        "lora": "models/finetune/exp015_npo_erased_lora10e2/epoch-1.safetensors",
    },
    "npo_base": {
        "base_model": "Wan-AI/Wan2.2-TI2V-5B",
        "lora": "models/finetune/exp015_npo_erased_lora10e2/epoch-1.safetensors",
    },
    "grad_ascent_erased": {
        "base_model": "wan5b/exp015_grad_ascent_erased",
        "lora": "models/finetune/exp015_grad_ascent_erased_lora10e2/epoch-1.safetensors",
    },
    "grad_ascent_base": {
        "base_model": "Wan-AI/Wan2.2-TI2V-5B",
        "lora": "models/finetune/exp015_grad_ascent_erased_lora10e2/epoch-1.safetensors",
    },
    "anchor_distill_erased": {
        "base_model": "wan5b/exp015_anchor_distill_erased",
        "lora": "models/finetune/exp015_anchor_distill_erased_lora10e2/epoch-1.safetensors",
    },
    "anchor_distill_base": {
        "base_model": "Wan-AI/Wan2.2-TI2V-5B",
        "lora": "models/finetune/exp015_anchor_distill_erased_lora10e2/epoch-1.safetensors",
    },
}


def quantize_model(arm: str, output_dir: Path):
    """
    量化单个臂的模型

    策略：
    1. 加载 base 模型（擦除后或原始）
    2. 加载微调 LoRA
    3. 合并 LoRA 到模型
    4. 量化 DiT（NF4）
    5. 保存量化后的完整模型
    """
    if arm not in ARMS:
        raise ValueError(f"未知的 arm: {arm}")

    arm_config = ARMS[arm]
    base_model_id = arm_config["base_model"]
    lora_path = arm_config["lora"]

    print(f"\n{'='*60}")
    print(f"量化: {arm}")
    print(f"{'='*60}")
    print(f"基座模型: {base_model_id}")
    print(f"LoRA: {lora_path}")
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

    # 构建模型配置（使用 model_id 方式，diffsynth 会自动处理分片）
    print("构建模型配置...")
    model_configs = [
        ModelConfig(model_id=base_model_id, origin_file_pattern=pattern)
        for pattern in MODEL_FILES
    ]

    # Tokenizer 配置
    tokenizer_config = ModelConfig(
        model_id="Wan-AI/Wan2.1-T2V-1.3B",
        origin_file_pattern="google/umt5-xxl/"
    )

    print("加载模型...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )

    # 加载 LoRA
    print(f"加载 LoRA: {lora_path}")
    pipe.load_lora(pipe.dit, lora_path, alpha=1.0)

    # 量化 DiT（包含 LoRA 层）
    print("量化 DiT (NF4)...")
    quant_config = QuantizeConfig(method="bitsandbytes_nf4")
    quant_config.quantize_model(pipe.dit, compute_device="cuda", model_device="cuda")

    # 保存量化模型
    print("保存量化模型...")

    # 保存 DiT
    dit_state_dict = pipe.dit.state_dict()
    dit_output = output_dir / "diffusion_pytorch_model.safetensors"
    save_file(dit_state_dict, dit_output)

    # 复制其他组件（T5, VAE）- 这些不量化
    import shutil
    from glob import glob

    # 找到源模型目录
    if base_model_id.startswith("wan5b/"):
        source_dir = Path("models") / base_model_id.replace("wan5b/", "")
    else:
        # 从 HF cache 找
        import os
        hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        # 简化：直接从 models 目录复制
        source_dir = Path("models") / base_model_id.split("/")[-1]

    if not source_dir.exists():
        print(f"⚠️  源目录不存在，跳过复制其他组件: {source_dir}")
    else:
        # 复制 T5 和 VAE
        for pattern in ["models_t5_umt5-xxl-enc-bf16.safetensors", "Wan2.2_VAE.safetensors"]:
            source_files = list(source_dir.glob(pattern))
            if source_files:
                for src in source_files:
                    dst = output_dir / src.name
                    if not dst.exists():
                        print(f"  复制: {src.name}")
                        shutil.copy2(src, dst)

    # 保存量化配置
    config = {
        "arm": arm,
        "base_model": base_model_id,
        "lora": lora_path,
        "quantization": "nf4",
        "quantized_component": "DiT",
    }

    with open(output_dir / "quant_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ 量化完成: {output_dir}")
    print(f"   DiT 大小: {dit_output.stat().st_size / 1024**3:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Exp016: 量化 Exp015 的 8 个模型")
    parser.add_argument("--arm", required=True, choices=list(ARMS.keys()),
                        help="要量化的臂")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="输出目录")

    args = parser.parse_args()

    quantize_model(args.arm, args.output_dir)


if __name__ == "__main__":
    main()
