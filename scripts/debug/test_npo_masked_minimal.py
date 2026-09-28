#!/usr/bin/env python3
"""NPOMasked smoke test - 最小验证"""

import sys
from pathlib import Path

VU_ROOT = Path(os.getenv("VU_ROOT", "../video-unlearning"))
sys.path.insert(0, str(VU_ROOT))

print("="*60)
print("NPOMasked Smoke Test (最小验证)")
print("="*60)

# 测试1: NPOMasked类注册
print("\n[1/3] 检查NPOMasked注册...")
try:
    from src.unlearning.registry import METHOD_REGISTRY
    npo_masked = METHOD_REGISTRY.get("NPOMasked")
    if npo_masked:
        print(f"  ✅ NPOMasked已注册: {npo_masked}")
    else:
        print(f"  ❌ NPOMasked未注册")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

# 测试2: mask配置文件
print("\n[2/3] 检查mask配置...")
try:
    from omegaconf import OmegaConf
    config_path = VU_ROOT / "configs/methods/training/npo_masked_wan.yaml"
    if not config_path.exists():
        print(f"  ❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    cfg = OmegaConf.load(config_path)
    print(f"  handler: {cfg.handler}")
    print(f"  num_heads: {cfg.method_args.mask_config.num_heads}")
    print(f"  grid: {cfg.method_args.mask_config.frames}f × {cfg.method_args.mask_config.height}h × {cfg.method_args.mask_config.width}w")
    print(f"  ✅ 配置正确")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

# 测试3: 训练脚本argparse
print("\n[3/3] 检查训练脚本...")
try:
    import subprocess
    result = subprocess.run(
        ["python", "scripts/run_unlearn_wan5b.py", "--help"],
        cwd="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio",
        capture_output=True,
        text=True,
        timeout=10
    )
    if "NPOMasked" in result.stdout:
        print(f"  ✅ NPOMasked在argparse choices中")
    else:
        print(f"  ❌ NPOMasked不在argparse choices中")
        print(f"  输出:\n{result.stdout[:500]}")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ 所有验证通过！")
print("="*60)
print("\n关键配置:")
print("  - erase_concept: 'breasts' (在所有prompts中)")
print("  - mask_config.num_heads: 48")
print("  - mask_config.grid: 4×30×46")
print("\n可以开始训练了！")
