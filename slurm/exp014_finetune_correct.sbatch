#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH -t 3-00:00:00
#SBATCH -J exp014-finetune
#SBATCH -o slurm/logs/exp014_finetune-%j.out
#SBATCH -e slurm/logs/exp014_finetune-%j.err

# ============================================================================
# Exp014 Stage 3: 微调（repeat=25, epochs=20）
#
# 目标：生成 20 个 checkpoint，用于细粒度分析"安全回潮曲线"
#
# 与 Exp012 的差异：
#   Exp012: repeat=10, epochs=2  → 20次曝光，2个checkpoint
#   Exp014: repeat=25, epochs=20 → 500次曝光，20个checkpoint
#
# 两臂：
#   - base_ft: models/wan5b/exp014_base + 微调
#   - erased_ft: models/wan5b/exp014_erased + 微调
# ============================================================================

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# ---- 环境配置 ----
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0  # 确保两臂数据顺序一致

mkdir -p slurm/logs

echo "=================================================="
echo "Exp014 Stage 3 - 微调（repeat=25, epochs=20）"
echo "=================================================="
echo "节点: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo ""
echo "配置:"
echo "  Dataset repeat: 25"
echo "  Epochs: 20"
echo "  Total exposure: 500次/样本"
echo "  Checkpoints: 20个（每epoch一个）"
echo "=================================================="
echo ""

# ---- 前置检查 ----
echo "🔍 前置检查:"
echo "-----------------------------------"

BASE_MODEL="models/wan5b/exp014_base"
ERASED_MODEL="models/wan5b/exp014_erased"

if [ ! -d "$BASE_MODEL" ]; then
    echo "❌ 错误: Base 模型不存在: $BASE_MODEL"
    echo "   请先运行 Stage 2（LoRA 合并）"
    exit 1
fi
echo "✓ Base 模型: $BASE_MODEL"

if [ ! -d "$ERASED_MODEL" ]; then
    echo "❌ 错误: Erased 模型不存在: $ERASED_MODEL"
    echo "   请先运行 Stage 2（LoRA 合并）"
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
echo "🔥 臂 1/2: Erased + 微调"
echo "-----------------------------------"

ERASED_OUTPUT="models/finetune/exp014_erased_ft"
mkdir -p "$ERASED_OUTPUT"

python -u scripts/finetune_wan5b.py \
    --arm erased \
    --dataset_repeat 25 \
    --num_epochs 20 \
    --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B-exp014-erased:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B-exp014-erased:models_t5_umt5-xxl-enc-bf16.safetensors,Wan-AI/Wan2.2-TI2V-5B-exp014-erased:Wan2.2_VAE.safetensors" \
    --output_path "$ERASED_OUTPUT" \
    --save_steps 9999999 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --use_gradient_checkpointing

if [ $? -ne 0 ]; then
    echo "❌ Erased 臂微调失败！"
    exit 1
fi

echo ""
echo "✅ Erased 臂完成"
echo "产物: $ERASED_OUTPUT"
ls -lh "$ERASED_OUTPUT/"
echo ""

# ---- 第二臂：Base + FT ----
echo "🔥 臂 2/2: Base + 微调"
echo "-----------------------------------"

BASE_OUTPUT="models/finetune/exp014_base_ft"
mkdir -p "$BASE_OUTPUT"

python -u scripts/finetune_wan5b.py \
    --arm base \
    --dataset_repeat 25 \
    --num_epochs 20 \
    --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B-exp014-base:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B-exp014-base:models_t5_umt5-xxl-enc-bf16.safetensors,Wan-AI/Wan2.2-TI2V-5B-exp014-base:Wan2.2_VAE.safetensors" \
    --output_path "$BASE_OUTPUT" \
    --save_steps 9999999 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --use_gradient_checkpointing

if [ $? -ne 0 ]; then
    echo "❌ Base 臂微调失败！"
    exit 1
fi

echo ""
echo "✅ Base 臂完成"
echo "产物: $BASE_OUTPUT"
ls -lh "$BASE_OUTPUT/"
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
        echo "  Checkpoint 数量: $checkpoint_count"
        echo "  预期: 20 个 (epoch-01 到 epoch-20)"

        if [ "$checkpoint_count" -eq 20 ]; then
            echo "  ✅ Checkpoint 数量正确"
        else
            echo "  ⚠️  Checkpoint 数量不足"
        fi

        echo "  目录内容:"
        ls -lh "$output_dir/" | head -10
    else
        echo "  ❌ 输出目录不存在"
    fi
done

echo ""
echo "=================================================="
echo "下一步: Stage 4（量化）或 Stage 5（生成评估）"
echo "=================================================="
