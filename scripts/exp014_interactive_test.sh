#!/bin/bash
# Exp014 交互式测试脚本
# 用于在 GPU 节点上逐阶段验证配置

set -e

echo "=================================================="
echo "Exp014 交互式测试"
echo "=================================================="
echo "节点: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "当前目录: $(pwd)"
echo ""

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

cd "$(dirname "$(dirname "$(readlink -f "$0")")")" # Auto: project root

# ============================================
# 阶段0: 环境检查
# ============================================
echo "🔍 阶段0: 环境检查"
echo "-----------------------------------"

echo "✓ 检查 Python 环境:"
which python
python --version

echo ""
echo "✓ 检查关键脚本:"
for script in run_unlearn_wan5b.py merge_unlearn_lora_wan5b.py finetune_wan5b.py; do
    if [ -f "scripts/$script" ]; then
        echo "  ✓ scripts/$script"
    else
        echo "  ✗ scripts/$script (缺失)"
    fi
done

echo ""
echo "✓ 检查基座模型:"
if [ -d "models/Wan-AI/Wan2.2-TI2V-5B-Diffusers" ]; then
    echo "  ✓ 5B Diffusers 基座存在"
else
    echo "  ✗ 5B Diffusers 基座缺失"
    exit 1
fi

echo ""
echo "✓ 检查缓存视频:"
if [ -d "data/wan5b/unlearn_baseline" ]; then
    forget_count=$(find data/wan5b/unlearn_baseline -name "*.mp4" -path "*/nudity_forget/*" | wc -l)
    retain_count=$(find data/wan5b/unlearn_baseline -name "*.mp4" -path "*/nudity_retain/*" | wc -l)
    echo "  ✓ 缓存视频: forget=$forget_count, retain=$retain_count"
else
    echo "  ✗ 缓存视频目录不存在"
    exit 1
fi

echo ""
echo "=================================================="
echo "环境检查通过！"
echo "=================================================="
echo ""

# ============================================
# 阶段1: 测试擦除训练（仅10步，快速验证）
# ============================================
echo "🔥 阶段1: 测试擦除训练（10步快速验证）"
echo "-----------------------------------"

TEST_DIR="models/unlearn/exp014_test_10steps"
mkdir -p "$TEST_DIR"

echo "运行命令:"
echo "python scripts/run_unlearn_wan5b.py \\"
echo "    --train \\"
echo "    --method GradAscent \\"
echo "    --erase-concept nudity \\"
echo "    --unlearn-steps 10 \\"
echo "    --save-every 5 \\"
echo "    --batch-size 1 \\"
echo "    --learning-rate 1e-5 \\"
echo "    --train-dir \"$TEST_DIR\""
echo ""

python scripts/run_unlearn_wan5b.py \
    --train \
    --method GradAscent \
    --erase-concept nudity \
    --unlearn-steps 10 \
    --save-every 5 \
    --batch-size 1 \
    --learning-rate 1e-5 \
    --train-dir "$TEST_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 阶段1成功！检查产物:"
    ls -lh "$TEST_DIR/"
    echo ""
else
    echo ""
    echo "❌ 阶段1失败！"
    exit 1
fi

# ============================================
# 阶段2: 测试 LoRA 合并
# ============================================
echo "🔗 阶段2: 测试 LoRA 合并"
echo "-----------------------------------"

MERGE_DIR="models/unlearn/exp014_test_merged"
mkdir -p "$MERGE_DIR"

echo "运行命令:"
echo "python scripts/merge_unlearn_lora_wan5b.py \\"
echo "    --adapter \"$TEST_DIR/adapter_final.pt\" \\"
echo "    --base-model-path \"models/Wan-AI/Wan2.2-TI2V-5B-Diffusers\" \\"
echo "    --output-dir \"$MERGE_DIR\" \\"
echo "    --bridge-output-dir \"models/Wan-AI/Wan2.2-TI2V-5B-erased-test\""
echo ""

# 检查脚本实际支持的参数
echo "检查 merge 脚本帮助信息..."
python scripts/merge_unlearn_lora_wan5b.py --help 2>&1 | head -30

echo ""
echo "=================================================="
echo "阶段1测试完成！"
echo ""
echo "如果看到以上帮助信息，说明脚本可以运行。"
echo "请检查参数名称是否正确，然后手动运行阶段2的合并命令。"
echo "=================================================="
