#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 16:00:00
#SBATCH -J exp014-stage2-3
#SBATCH -o logs/exp014_stage2_3_%j.out
#SBATCH -e logs/exp014_stage2_3_%j.err

# Exp 014 阶段2+3测试：合并 + 微调
# 验证参数修复是否正确

set -e

echo "=================================================="
echo "Exp 014 - Stage 2+3 Test"
echo "=================================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "=================================================="
echo ""

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

mkdir -p logs

# ============================================
# 阶段2: LoRA合并
# ============================================
echo "🔗 阶段2: LoRA合并（创建base & erased模型）"
echo ""

# 合并最终checkpoint (step 600) - 生成 erased 模型
# 使用本地已有的 base 模型
python scripts/merge_unlearn_lora_wan5b.py \
    --adapter "models/unlearn/exp014_wan5b_nudity_600steps/adapter_final.pt" \
    --base-dir "models/Wan-AI/Wan2.2-TI2V-5B" \
    --output-dir "models/unlearn/exp014_wan5b_nudity_600steps/merged" \
    --bridge-output-dir "models/Wan-AI/Wan2.2-TI2V-5B-erased" \
    --force

if [ $? -ne 0 ]; then
    echo "❌ LoRA合并失败！"
    exit 1
fi

echo "✓ base模型已存在: models/Wan-AI/Wan2.2-TI2V-5B"
echo "✓ erased模型已创建: models/Wan-AI/Wan2.2-TI2V-5B-erased"
echo "✅ 阶段2完成: base & erased模型已创建"
echo ""

# ============================================
# 阶段3: 微调（repeat=25, epochs=20）
# ============================================
echo "🎯 阶段3: 微调两臂（20个epoch，500次曝光/样本）"
echo ""

# 臂1: base微调
echo "  微调臂1: base..."
python scripts/finetune_wan5b.py \
    --arm base \
    --dataset_repeat 25 \
    --num_epochs 20 \
    --lora_rank 32 \
    --learning_rate 1e-4 \
    --output_path "models/finetune/exp014_base_ft" \
    --save_steps 0

if [ $? -ne 0 ]; then
    echo "❌ base微调失败！"
    exit 1
fi

# 臂2: erased微调
echo "  微调臂2: erased..."
python scripts/finetune_wan5b.py \
    --arm erased \
    --dataset_repeat 25 \
    --num_epochs 20 \
    --lora_rank 32 \
    --learning_rate 1e-4 \
    --output_path "models/finetune/exp014_erased_ft" \
    --save_steps 0

if [ $? -ne 0 ]; then
    echo "❌ erased微调失败！"
    exit 1
fi

echo "✅ 阶段3完成: 两臂微调完成（各20个checkpoint）"
echo ""
echo "=================================================="
echo "✅ 阶段2+3测试完成"
echo "完成时间: $(date)"
echo "=================================================="
