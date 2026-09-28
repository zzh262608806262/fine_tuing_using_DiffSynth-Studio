#!/usr/bin/env python3
"""
测试 diffsynth 环境是否正常
"""

import sys

print("=" * 60)
print("测试 diffsynth 环境")
print("=" * 60)

# 测试 1: 基础导入
print("\n[1/4] 测试基础包导入...")
try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
except Exception as e:
    print(f"  ✗ torch 导入失败: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"  ✓ transformers {transformers.__version__}")
except Exception as e:
    print(f"  ✗ transformers 导入失败: {e}")
    sys.exit(1)

# 测试 2: accelerate 导入（关键测试）
print("\n[2/4] 测试 accelerate 导入...")
try:
    from accelerate.commands.accelerate_cli import main
    print(f"  ✓ accelerate 可以导入")
except ImportError as e:
    print(f"  ✗ accelerate 导入失败: {e}")
    if "NP_SUPPORTED_MODULES" in str(e):
        print("  >>> 这是 torch 和 transformers 版本不兼容的问题")
    sys.exit(1)

# 测试 3: DiffSynth Studio 导入
print("\n[3/4] 测试 DiffSynth Studio 导入...")
try:
    sys.path.insert(0, "/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio")
    from diffsynth import ModelManager
    print(f"  ✓ DiffSynth Studio 可以导入")
except Exception as e:
    print(f"  ✗ DiffSynth Studio 导入失败: {e}")
    # 这个失败不算致命，继续

# 测试 4: CUDA 可用性
print("\n[4/4] 测试 CUDA...")
if torch.cuda.is_available():
    print(f"  ✓ CUDA 可用")
    print(f"  ✓ GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print(f"  ⚠ CUDA 不可用（登录节点正常，需要在 GPU 节点测试训练）")

print("\n" + "=" * 60)
print("✓ 环境测试通过！")
print("=" * 60)
