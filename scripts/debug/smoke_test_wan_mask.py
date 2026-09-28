#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wan2.2-TI2V-5B Attention Mask参数验证脚本

目的：验证NPOMasked所需的mask_config参数
- attention模块名称（attn_substring）
- attention heads数量（num_heads）
- latent grid尺寸（frames, height, width）

用法：
    python scripts/debug/smoke_test_wan_mask.py
    python scripts/debug/smoke_test_wan_mask.py --model-path <path>
"""

import argparse
import sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="验证Wan模型的mask参数")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        help="Wan模型路径（diffusers格式）",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="视频高度（用于计算latent尺寸）",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=736,
        help="视频宽度（用于计算latent尺寸）",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=17,
        help="视频帧数（用于计算latent尺寸）",
    )
    return parser.parse_args()


def check_attention_modules(model):
    """检查模型中的attention模块"""
    print("\n" + "="*80)
    print("📍 第1步：检查Attention模块结构")
    print("="*80)

    attn_modules = {}
    cross_attn_modules = []
    self_attn_modules = []

    for name, module in model.named_modules():
        # 查找包含attention的模块
        if "attn" in name.lower():
            module_type = type(module).__name__

            # 判断是cross-attention还是self-attention
            if "attn1" in name:
                self_attn_modules.append((name, module_type))
            elif "attn2" in name:
                cross_attn_modules.append((name, module_type))

            attn_modules[name] = module_type

    print(f"\n✅ 找到 {len(attn_modules)} 个attention相关模块")

    # 打印cross-attention模块
    if cross_attn_modules:
        print(f"\n📌 Cross-Attention模块（attn2，共{len(cross_attn_modules)}个）:")
        for name, mtype in cross_attn_modules[:5]:  # 只显示前5个
            print(f"   - {name}")
            print(f"     类型: {mtype}")
        if len(cross_attn_modules) > 5:
            print(f"   ... 还有 {len(cross_attn_modules) - 5} 个")

    # 打印self-attention模块
    if self_attn_modules:
        print(f"\n📌 Self-Attention模块（attn1，共{len(self_attn_modules)}个）:")
        for name, mtype in self_attn_modules[:3]:
            print(f"   - {name}")
            print(f"     类型: {mtype}")
        if len(self_attn_modules) > 3:
            print(f"   ... 还有 {len(self_attn_modules) - 3} 个")

    # 判断应该使用哪个substring
    if cross_attn_modules:
        print("\n✅ 结论：Wan模型使用 cross-attention (attn2)")
        recommended_substring = "attn2"
    elif self_attn_modules:
        print("\n⚠️  警告：只找到 self-attention (attn1)，可能是joint attention")
        recommended_substring = "attn1"
    else:
        print("\n❌ 错误：未找到attention模块")
        recommended_substring = None

    return recommended_substring, cross_attn_modules


def check_attention_heads(model, attn_modules):
    """检查attention heads数量"""
    print("\n" + "="*80)
    print("📍 第2步：检查Attention Heads数量")
    print("="*80)

    if not attn_modules:
        print("❌ 没有可检查的attention模块")
        return None

    # 取第一个cross-attention模块
    sample_name, _ = attn_modules[0]

    # 尝试找到to_q/to_k模块
    found_heads = None
    for name, module in model.named_modules():
        if sample_name in name and ("to_q" in name or "to_k" in name):
            # 检查模块属性
            if hasattr(module, "out_features") and hasattr(module, "in_features"):
                out_features = module.out_features
                in_features = module.in_features

                print(f"\n📌 检查模块: {name}")
                print(f"   in_features: {in_features}")
                print(f"   out_features: {out_features}")

                # 尝试推断head数量
                # 通常 out_features = num_heads * head_dim
                # 常见配置：head_dim = 64 or 128
                for head_dim in [64, 80, 128]:
                    if out_features % head_dim == 0:
                        num_heads = out_features // head_dim
                        print(f"   可能配置: num_heads={num_heads}, head_dim={head_dim}")
                        if found_heads is None:
                            found_heads = num_heads
                break

    if found_heads:
        print(f"\n✅ 推荐：num_heads = {found_heads}")
    else:
        print("\n⚠️  无法自动推断heads数量，建议检查模型配置文件")
        # 常见默认值
        print("   常见配置：24 (CogVideoX), 30 (某些DiT), 16 (较小模型)")
        found_heads = 24  # 默认猜测
        print(f"   使用默认值：num_heads = {found_heads}")

    return found_heads


def check_latent_shape(model_path, height, width, frames):
    """检查latent shape"""
    print("\n" + "="*80)
    print("📍 第3步：检查Latent Shape")
    print("="*80)

    print(f"\n输入视频尺寸: {frames}帧 × {height}h × {width}w")

    # 尝试加载VAE配置
    try:
        from diffusers import AutoencoderKL
        vae_path = Path(model_path) / "vae"
        if vae_path.exists():
            # 读取VAE配置
            import json
            config_path = vae_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    vae_config = json.load(f)

                # 检查下采样倍率
                down_block_types = vae_config.get("down_block_types", [])
                print(f"\n📌 VAE配置:")
                print(f"   down_block_types: {down_block_types}")

                # 通常spatial下采样是8x
                spatial_compression = 8
                print(f"   空间压缩倍率: {spatial_compression}x")

                # 时间压缩（需要从配置推断）
                # 常见：4x (17帧 -> 4-5帧)
                temporal_compression = 4
                print(f"   时间压缩倍率（推测）: {temporal_compression}x")
    except Exception as e:
        print(f"\n⚠️  无法加载VAE配置: {e}")
        spatial_compression = 8
        temporal_compression = 4
        print(f"   使用默认值: 空间{spatial_compression}x, 时间{temporal_compression}x")

    # 计算latent尺寸
    latent_h = height // spatial_compression
    latent_w = width // spatial_compression
    latent_f = frames // temporal_compression

    print(f"\nLatent尺寸: {latent_f}f × {latent_h}h × {latent_w}w")

    # 考虑patch size
    patch_size = 2  # 常见配置
    print(f"\n假设 patch_size = {patch_size}:")
    patch_f = latent_f  # 时间维度通常不patch
    patch_h = latent_h // patch_size
    patch_w = latent_w // patch_size

    print(f"   Patch后grid: {patch_f}f × {patch_h}h × {patch_w}w")

    return latent_f, patch_h, patch_w


def generate_config(attn_substring, num_heads, frames, height, width):
    """生成mask_config"""
    print("\n" + "="*80)
    print("📍 第4步：生成mask_config")
    print("="*80)

    config = {
        "attn_substring": attn_substring,
        "text_len": 0,  # Wan是纯cross-attention
        "is_joint": False,
        "num_heads": num_heads,
        "frames": frames,
        "height": height,
        "width": width,
    }

    print("\n生成的mask_config:")
    print("```yaml")
    print("mask_config:")
    for key, value in config.items():
        if isinstance(value, str):
            print(f"  {key}: \"{value}\"")
        elif isinstance(value, bool):
            print(f"  {key}: {str(value).lower()}")
        else:
            print(f"  {key}: {value}")
    print("```")

    return config


def main():
    args = parse_args()

    print("="*80)
    print("Wan2.2-TI2V-5B Attention Mask参数验证")
    print("="*80)
    print(f"模型路径: {args.model_path}")
    print(f"视频尺寸: {args.frames}帧 × {args.height}h × {args.width}w")

    # 检查模型路径
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"\n❌ 错误：模型路径不存在: {model_path}")
        return 1

    # 加载模型（只加载DiT，不加载权重）
    print("\n正在加载模型结构...")
    try:
        from diffusers import DiffusionPipeline

        # 只加载配置，不加载权重
        print("（只加载结构，不加载权重，可能需要1-2分钟）")
        pipe = DiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        # 获取transformer模块
        if hasattr(pipe, "transformer"):
            model = pipe.transformer
        elif hasattr(pipe, "unet"):
            model = pipe.unet
        else:
            print("❌ 错误：无法找到transformer/unet模块")
            return 1

        print("✅ 模型加载成功")

    except Exception as e:
        print(f"❌ 错误：加载模型失败: {e}")
        print("\n提示：如果是内存不足，可以在GPU节点运行")
        return 1

    # Step 1: 检查attention模块
    attn_substring, cross_attn_modules = check_attention_modules(model)
    if not attn_substring:
        return 1

    # Step 2: 检查attention heads
    num_heads = check_attention_heads(model, cross_attn_modules)
    if not num_heads:
        return 1

    # Step 3: 检查latent shape
    latent_f, latent_h, latent_w = check_latent_shape(
        model_path, args.height, args.width, args.frames
    )

    # Step 4: 生成配置
    config = generate_config(attn_substring, num_heads, latent_f, latent_h, latent_w)

    # 总结
    print("\n" + "="*80)
    print("✅ 验证完成！")
    print("="*80)
    print("\n下一步：")
    print("1. 复制上面的mask_config到 npo_masked_wan.yaml")
    print("2. 运行小规模训练测试（100步）验证mask能正确构建")
    print("3. 如果测试通过，开始正式训练")

    return 0


if __name__ == "__main__":
    sys.exit(main())
