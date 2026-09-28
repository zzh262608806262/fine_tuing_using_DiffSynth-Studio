#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 16:00:00
#SBATCH -J exp014-finetune
#SBATCH -o logs/exp014_finetune_%j.out
#SBATCH -e logs/exp014_finetune_%j.err

# Exp 014: 微调两臂（base + erased）
# 配置: repeat=25, epochs=20 (总曝光500次，20个checkpoint)

set -e

echo "=================================================="
echo "Exp 014 - Finetune (base & erased)"
echo "=================================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo ""

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 配置
BASE_MODEL="models/wan5b/exp014_base"
ERASED_MODEL="models/wan5b/exp014_erased"
DATASET_META="data/tiger_dataset/metadata_100.csv"
DATASET_REPEAT=25         # 小repeat
NUM_EPOCHS=20             # 大epoch（20个checkpoint）
LORA_RANK=32
LEARNING_RATE=1e-4

echo "📊 微调配置:"
echo "  - Dataset repeat: $DATASET_REPEAT"
echo "  - Num epochs: $NUM_EPOCHS"
echo "  - 总曝光: $((DATASET_REPEAT * NUM_EPOCHS)) 次/样本"
echo "  - LoRA rank: $LORA_RANK"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Checkpoint数: $NUM_EPOCHS 个"
echo ""

# 臂1: base微调
echo "🚀 [1/2] 微调base模型..."
python scripts/finetune_wan5b.py \
    --arm base \
    --dataset_metadata_path "$DATASET_META" \
    --dataset_repeat $DATASET_REPEAT \
    --num_epochs $NUM_EPOCHS \
    --lora_rank $LORA_RANK \
    --learning_rate $LEARNING_RATE \
    --output_path "models/finetune/exp014_base_ft" \
    --use_gradient_checkpointing

echo "✅ base微调完成"
echo ""

# 臂2: erased微调
echo "🚀 [2/2] 微调erased模型..."
python scripts/finetune_wan5b.py \
    --arm erased \
    --dataset_metadata_path "$DATASET_META" \
    --dataset_repeat $DATASET_REPEAT \
    --num_epochs $NUM_EPOCHS \
    --lora_rank $LORA_RANK \
    --learning_rate $LEARNING_RATE \
    --output_path "models/finetune/exp014_erased_ft" \
    --use_gradient_checkpointing

echo "✅ erased微调完成"
echo ""

echo "=================================================="
echo "✅ Exp 014微调阶段完成！"
echo "完成时间: $(date)"
echo ""
echo "产物："
echo "  - Base微调: models/finetune/exp014_base_ft/ (20个epoch)"
echo "  - Erased微调: models/finetune/exp014_erased_ft/ (20个epoch)"
echo ""
echo "下一步: 运行量化脚本"
echo "  sbatch slurm/exp014_wan5b_quantize.sbatch"
echo "=================================================="
