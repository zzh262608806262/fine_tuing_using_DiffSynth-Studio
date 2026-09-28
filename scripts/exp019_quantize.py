#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp019 Phase 9: 量化GradDiff擦除模型

使用bitsandbytes int8量化，减少显存占用同时保持安全性
"""
import argparse
import os
import sys
import torch
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def quantize_model(input_dir: str, output_dir: str, dtype: str = "int8"):
    """量化模型"""
    import bitsandbytes as bnb
    from safetensors.torch import load_file, save_file
    import glob

    print(f"量化模型: {input_dir} -> {output_dir}")
    print(f"量化类型: {dtype}")

    os.makedirs(output_dir, exist_ok=True)

    # 查找所有safetensors文件
    safetensor_files = glob.glob(os.path.join(input_dir, "*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {input_dir}")

    print(f"Found {len(safetensor_files)} safetensor files")

    total_original_size = 0
    total_quantized_size = 0

    for shard_file in safetensor_files:
        shard_name = os.path.basename(shard_file)
        print(f"\nProcessing: {shard_name}")

        # 加载原始权重
        state_dict = load_file(shard_file)
        print(f"  Loaded {len(state_dict)} tensors")

        # 量化
        quantized_dict = {}
        for key, tensor in state_dict.items():
            original_size = tensor.numel() * tensor.element_size()
            total_original_size += original_size

            if tensor.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                # 量化到int8
                if dtype == "int8":
                    # 简单量化：转为float32 -> 量化scale -> int8
                    tensor_fp32 = tensor.float()
                    scale = tensor_fp32.abs().max() / 127.0
                    quantized = torch.clamp(tensor_fp32 / scale, -128, 127).to(torch.int8)

                    # 存储量化tensor和scale
                    quantized_dict[key] = quantized
                    quantized_dict[f"{key}.scale"] = scale

                    quantized_size = quantized.numel() + scale.numel() * scale.element_size()
                else:
                    quantized_dict[key] = tensor
                    quantized_size = original_size
            else:
                # 非浮点tensor保持不变
                quantized_dict[key] = tensor
                quantized_size = original_size

            total_quantized_size += quantized_size

        # 保存量化后的分片
        output_file = os.path.join(output_dir, shard_name)
        save_file(quantized_dict, output_file)
        print(f"  Saved: {output_file}")

    # 复制config文件
    for config_file in ["config.json", "model_index.json", "diffusers_config.json"]:
        src = os.path.join(input_dir, config_file)
        if os.path.exists(src):
            import shutil
            dst = os.path.join(output_dir, config_file)
            shutil.copy2(src, dst)
            print(f"Copied: {config_file}")

    # 统计
    compression_ratio = total_original_size / total_quantized_size
    print(f"\n量化完成:")
    print(f"  原始大小: {total_original_size / 1e9:.2f} GB")
    print(f"  量化大小: {total_quantized_size / 1e9:.2f} GB")
    print(f"  压缩比: {compression_ratio:.2f}x")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Exp019 模型量化")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="输入模型目录")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出量化模型目录")
    parser.add_argument("--dtype", type=str, default="int8",
                        choices=["int8", "int4"],
                        help="量化精度")
    args = parser.parse_args()

    quantize_model(args.input_dir, args.output_dir, args.dtype)


if __name__ == "__main__":
    main()
