#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exp017 蒸馏模型测试生成脚本

测试5个蒸馏模型（4步推理）的生成质量
在交互节点上快速验证，成功后再批量生成

用法：
    # 在交互节点测试
    python scripts/exp017_test_distill_generation.py --arm base --test-prompt "A tiger walking in the forest"

    # 测试所有模型
    python scripts/exp017_test_distill_generation.py --arm all --test-prompt "A tiger walking in the forest"
"""
import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video

# 5个蒸馏模型定义
DISTILL_MODELS = {
    "base": "models/train/exp017_base_distill/epoch-9.safetensors",
    "grad_ascent": "models/train/exp017_grad_ascent_distill/epoch-9.safetensors",
    "esd": "models/train/exp017_esd_distill/epoch-9.safetensors",
    "npo": "models/train/exp017_npo_distill/epoch-9.safetensors",
    "anchor_distill": "models/train/exp017_anchor_distill_distill/epoch-9.safetensors",
}

def test_single_model(arm: str, prompt: str, output_dir: str):
    """测试单个蒸馏模型"""
    print(f"\n{'='*60}")
    print(f"测试蒸馏模型: {arm}")
    print(f"{'='*60}\n")

    # 检查模型文件
    distill_path = DISTILL_MODELS[arm]
    if not os.path.exists(distill_path):
        print(f"❌ 错误: 模型文件不存在: {distill_path}")
        return False

    print(f"✓ 模型文件: {distill_path}")
    print(f"✓ 推理步数: 4步（蒸馏目标）")
    print(f"✓ 提示词: {prompt}")

    try:
        # 加载模型
        print("\n[1/3] 加载模型...")
        start_time = time.time()

        import torch
        from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(path=distill_path),  # 蒸馏后的DiT
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.safetensors"),
                ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.safetensors"),
            ],
            tokenizer_config=ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B",
                origin_file_pattern="google/umt5-xxl/"
            ),
        )

        load_time = time.time() - start_time
        print(f"✓ 模型加载完成 ({load_time:.1f}秒)")

        # 生成视频
        print("\n[2/3] 生成视频（4步推理）...")
        gen_start = time.time()

        video = pipe(
            prompt=prompt,
            num_inference_steps=4,  # 蒸馏模型目标步数
            cfg_scale=1.0,           # 蒸馏通常用1.0
            height=480,
            width=736,
            num_frames=17,
            seed=42,
        )

        gen_time = time.time() - gen_start
        print(f"✓ 视频生成完成 ({gen_time:.1f}秒)")

        # 保存视频
        print("\n[3/3] 保存视频...")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{arm}_test.mp4")
        save_video(video, output_path, fps=8, quality=5)

        print(f"✓ 视频已保存: {output_path}")
        print(f"\n总耗时: {time.time() - start_time:.1f}秒")
        print(f"  - 加载: {load_time:.1f}秒")
        print(f"  - 生成: {gen_time:.1f}秒")

        return True

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Exp017蒸馏模型测试生成")
    parser.add_argument("--arm", required=True,
                       choices=list(DISTILL_MODELS.keys()) + ["all"],
                       help="要测试的模型臂")
    parser.add_argument("--test-prompt", default="A tiger walking in the forest",
                       help="测试提示词")
    parser.add_argument("--output-dir", default="outputs/exp017_test",
                       help="输出目录")

    args = parser.parse_args()

    print("=" * 60)
    print("Exp017 蒸馏模型测试生成")
    print("=" * 60)
    print(f"工作目录: {_ROOT}")
    print(f"输出目录: {args.output_dir}")
    print(f"测试提示词: {args.test_prompt}")

    if args.arm == "all":
        print(f"\n将测试所有5个模型")
        arms_to_test = list(DISTILL_MODELS.keys())
    else:
        arms_to_test = [args.arm]

    # 测试模型
    results = {}
    for arm in arms_to_test:
        success = test_single_model(arm, args.test_prompt, args.output_dir)
        results[arm] = success

    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    for arm, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{arm:20s} {status}")

    success_count = sum(results.values())
    total_count = len(results)
    print(f"\n成功: {success_count}/{total_count}")

    if success_count == total_count:
        print("\n🎉 所有模型测试成功！可以开始批量生成。")
    else:
        print("\n⚠️ 部分模型测试失败，请检查错误日志。")

if __name__ == "__main__":
    main()
