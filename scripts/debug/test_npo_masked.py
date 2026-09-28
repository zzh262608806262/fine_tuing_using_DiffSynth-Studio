#!/usr/bin/env python3
"""NPOMasked smoke test - 验证masked方法能否正常运行"""

import sys
import torch
from pathlib import Path

# 设置VU路径
VU_ROOT = Path(os.getenv("VU_ROOT", "../video-unlearning"))
sys.path.insert(0, str(VU_ROOT))

print("="*60)
print("NPOMasked Smoke Test")
print("="*60)

# 测试1: 验证NPOMasked可以导入
print("\n[1/4] 导入NPOMasked...")
try:
    from src.unlearning.registry import METHOD_REGISTRY
    npo_masked_class = METHOD_REGISTRY.get("NPOMasked")
    if npo_masked_class:
        print(f"  ✅ NPOMasked类: {npo_masked_class}")
    else:
        print(f"  ❌ NPOMasked未在registry中注册")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ 导入失败: {e}")
    sys.exit(1)

# 测试2: 验证concept token定位
print("\n[2/4] 测试concept token定位...")
try:
    from src.unlearning.training.attention_mask import locate_concept_tokens
    from transformers import AutoTokenizer

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers/text_encoder"
    )

    # 测试prompt（来自实际数据）
    test_prompt = "A documentary-style video of an adult woman standing completely still standing in an empty bathtub, facing the camera, framed from the waist up, her bare breasts clearly visible; neutral expression, no sexual activity, static camera, she stays fully in frame for the entire video."

    # 测试concept="breasts"（应该成功）
    try:
        positions = locate_concept_tokens(tokenizer, test_prompt, "breasts", max_length=256)
        print(f"  ✅ concept='breasts' 找到位置: {positions.tolist()}")
    except ValueError as e:
        print(f"  ❌ concept='breasts' 失败: {e}")
        sys.exit(1)

    # 测试concept="nudity"（应该失败）
    try:
        positions = locate_concept_tokens(tokenizer, test_prompt, "nudity", max_length=256)
        print(f"  ⚠️  concept='nudity' 意外成功: {positions.tolist()}")
    except ValueError as e:
        print(f"  ✅ concept='nudity' 正确失败: {e}")

except Exception as e:
    print(f"  ❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 验证mask配置
print("\n[3/4] 验证mask配置...")
try:
    from omegaconf import OmegaConf
    config_path = VU_ROOT / "configs/methods/training/npo_masked_wan.yaml"
    cfg = OmegaConf.load(config_path)

    print(f"  handler: {cfg.handler}")
    print(f"  num_heads: {cfg.method_args.mask_config.num_heads}")
    print(f"  grid: {cfg.method_args.mask_config.frames}f × {cfg.method_args.mask_config.height}h × {cfg.method_args.mask_config.width}w")
    print(f"  attn_substring: {cfg.method_args.mask_config.attn_substring}")
    print(f"  ✅ 配置加载成功")
except Exception as e:
    print(f"  ❌ 配置加载失败: {e}")
    sys.exit(1)

# 测试4: 验证训练脚本参数
print("\n[4/4] 验证训练脚本...")
try:
    import subprocess
    result = subprocess.run(
        ["python", "scripts/run_unlearn_wan5b.py", "--method", "NPOMasked", "--help"],
        cwd="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio",
        capture_output=True,
        text=True,
        timeout=10
    )
    if "NPOMasked" in result.stdout:
        print(f"  ✅ NPOMasked已在argparse choices中")
    else:
        print(f"  ❌ NPOMasked不在argparse choices中")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ 脚本验证失败: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ 所有测试通过！")
print("="*60)
print("\n建议:")
print("  - 使用 erase_concept='breasts' (出现在所有27条prompts)")
print("  - 或使用 erase_concept='bare' (出现在21条prompts)")
print("  - 不要使用 erase_concept='nudity' (不在prompts中)")
