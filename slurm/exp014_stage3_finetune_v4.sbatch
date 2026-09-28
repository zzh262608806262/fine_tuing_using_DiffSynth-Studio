#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 3-00:00:00
#SBATCH -J exp014-ft-v4
#SBATCH -o slurm/logs/exp014_finetune_v4-%j.out
#SBATCH -e slurm/logs/exp014_finetune_v4-%j.err

# ============================================================================
# Exp014 Stage 3: 微调 v4 - 移除 tensorboard 依赖
#
# 错误历史:
#   v1: pandas/numpy 二进制不兼容
#   v2: GLIBCXX_3.4.29 缺失
#   v3: 缺少 tensorboard 模块
#   v4: 移除 --enable_tensorboard_log 参数
# ============================================================================

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# ---- 环境配置 ----
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

# 使用 conda 的 C++ 库
export LD_LIBRARY_PATH="/home/x_jiage/.conda/envs/diffsynth/lib:$LD_LIBRARY_PATH"

export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0

mkdir -p slurm/logs

echo "=================================================="
echo "Exp014 Stage 3 v4 - 微调（repeat=25, epochs=20）"
echo "=================================================="
echo "节点: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo ""
echo "配置:"
echo "  Dataset repeat: 25"
echo "  Epochs: 20"
echo "  注意: 禁用 tensorboard 日志"
echo "=================================================="
echo ""

# ---- 前置检查 ----
echo "🔍 前置检查:"
echo "-----------------------------------"

BASE_MODEL="models/Wan-AI/Wan2.2-TI2V-5B"
ERASED_MODEL="models/Wan-AI/Wan2.2-TI2V-5B-erased"

if [ ! -d "$BASE_MODEL" ]; then
    echo "❌ 错误: Base 模型不存在: $BASE_MODEL"
    exit 1
fi
echo "✓ Base 模型: $BASE_MODEL"

if [ ! -d "$ERASED_MODEL" ]; then
    echo "❌ 错误: Erased 模型不存在: $ERASED_MODEL"
    exit 1
fi
echo "✓ Erased 模型: $ERASED_MODEL"

DATA_CSV="data/tiger_dataset/metadata_100.csv"
if [ ! -f "$DATA_CSV" ]; then
    echo "❌ 错误: 数据集不存在: $DATA_CSV"
    exit 1
fi
echo "✓ 数据集: $DATA_CSV"

echo ""
echo "=================================================="
echo "开始微调..."
echo "=================================================="
echo ""

# ---- 第一臂：Erased + FT ----
echo "🔥 臂 1/2: Erased + 微调 (预计 12-16 小时)"
echo "-----------------------------------"

ERASED_OUTPUT="models/finetune/exp014_erased_ft"
mkdir -p "$ERASED_OUTPUT"

# 关键修改：移除 --enable_tensorboard_log
python -u scripts/finetune_wan5b.py \
    --arm erased \
    --dataset_repeat 25 \
    --num_epochs 20 \
    --output_path "$ERASED_OUTPUT" \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --use_gradient_checkpointing \
    --no_tensorboard_log

if [ $? -ne 0 ]; then
    echo "❌ Erased 臂微调失败！"
    exit 1
fi

echo ""
echo "✅ Erased 臂完成"
echo ""

# ---- 第二臂：Base + FT ----
echo "🔥 臂 2/2: Base + 微调 (预计 12-16 小时)"
echo "-----------------------------------"

BASE_OUTPUT="models/finetune/exp014_base_ft"
mkdir -p "$BASE_OUTPUT"

python -u scripts/finetune_wan5b.py \
    --arm base \
    --dataset_repeat 25 \
    --num_epochs 20 \
    --output_path "$BASE_OUTPUT" \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --use_gradient_checkpointing \
    --no_tensorboard_log

if [ $? -ne 0 ]; then
    echo "❌ Base 臂微调失败！"
    exit 1
fi

echo ""
echo "✅ Base 臂完成"
echo ""

# ---- 产物检查 ----
echo "=================================================="
echo "✅ Stage 3 完成！"
echo "=================================================="
echo ""
echo "📦 产物检查:"
echo "-----------------------------------"

for arm in erased base; do
    output_dir="models/finetune/exp014_${arm}_ft"
    echo ""
    echo "臂: $arm"

    if [ -d "$output_dir" ]; then
        checkpoint_count=$(ls -1 "$output_dir"/epoch-*.safetensors 2>/dev/null | wc -l)
        total_size=$(du -sh "$output_dir" 2>/dev/null | cut -f1)

        echo "  路径: $output_dir"
        echo "  Checkpoint 数量: $checkpoint_count / 20"
        echo "  总大小: $total_size"

        if [ "$checkpoint_count" -eq 20 ]; then
            echo "  ✅ Checkpoint 数量正确"
        else
            echo "  ⚠️  Checkpoint 数量: $checkpoint_count (预期 20)"
        fi
    else
        echo "  ❌ 输出目录不存在: $output_dir"
    fi
done

echo ""
echo "=================================================="
echo "完成时间: $(date)"
echo "=================================================="
