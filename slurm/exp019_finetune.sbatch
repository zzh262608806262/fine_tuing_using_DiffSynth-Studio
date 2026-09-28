#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 3-00:00:00
#SBATCH -J exp019_ft_graddiff
#SBATCH -o outputs/exp019_finetune_%j.log
#SBATCH -e outputs/exp019_finetune_%j.err

# ============================================================================
# Exp019 Phase 5: GradDiff擦除模型微调（repeat=25, epochs=20）
#
# 目的: 验证GradDiff的retain约束是否减少微调回潮
# 对比: Exp015的GradAscent（无retain约束）
# ============================================================================

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "=================================================="
echo "Exp019 Phase 5 - GradDiff擦除模型微调"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "=================================================="

# ---- 环境配置 ----
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

export LD_LIBRARY_PATH="/home/x_jiage/.conda/envs/diffsynth/lib:$LD_LIBRARY_PATH"
export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0

echo ""
echo "Environment:"
python --version
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ---- 路径配置 ----
ERASED_DIT="models/wan5b/exp019_graddiff_erased"
BASE_MODEL="models/Wan-AI/Wan2.2-TI2V-5B"
OUTPUT_PATH="models/finetune/exp019_graddiff_erased_lora"
DATASET_PATH="data/tiger_dataset/metadata_100.csv"

echo "Configuration:"
echo "  Erased DiT: $ERASED_DIT"
echo "  Base (T5/VAE): $BASE_MODEL"
echo "  Output: $OUTPUT_PATH"
echo "  Dataset: $DATASET_PATH"
echo "  Dataset repeat: 25"
echo "  Epochs: 20"
echo "  LoRA rank: 32"
echo "  Checkpoints: 20 个（每 epoch 保存）"
echo "=================================================="
echo ""

# ---- 前置检查 ----
echo "🔍 Checking prerequisites..."

# 检查DiT文件
dit_files=(
    "diffusion_pytorch_model-00001-of-00003.safetensors"
    "diffusion_pytorch_model-00002-of-00003.safetensors"
    "diffusion_pytorch_model-00003-of-00003.safetensors"
)

for file in "${dit_files[@]}"; do
    if [ ! -f "$ERASED_DIT/$file" ]; then
        echo "❌ Error: Missing DiT file: $ERASED_DIT/$file"
        exit 1
    fi
done

# 检查T5/VAE文件
auxiliary_files=(
    "models_t5_umt5-xxl-enc-bf16.safetensors"
    "Wan2.2_VAE.safetensors"
)

for file in "${auxiliary_files[@]}"; do
    if [ ! -f "$BASE_MODEL/$file" ] && [ ! -L "$BASE_MODEL/$file" ]; then
        echo "❌ Error: Missing T5/VAE file: $BASE_MODEL/$file"
        exit 1
    fi
done

if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset not found: $DATASET_PATH"
    exit 1
fi

echo "✓ All prerequisites OK"
echo ""

# ---- 构建模型路径列表 ----
MODEL_PATHS=""
# DiT分片
for file in "${dit_files[@]}"; do
    MODEL_PATHS="${MODEL_PATHS}${ERASED_DIT}/${file},"
done
# T5和VAE
for file in "${auxiliary_files[@]}"; do
    MODEL_PATHS="${MODEL_PATHS}${BASE_MODEL}/${file},"
done
MODEL_PATHS="${MODEL_PATHS%,}"  # 移除末尾逗号

echo "Model paths:"
echo "$MODEL_PATHS" | tr ',' '\n' | sed 's/^/  /'
echo ""

# ---- 微调训练 ----
echo "🚀 Starting fine-tuning..."
echo "=================================================="

accelerate launch --mixed_precision=bf16 --num_processes=1 \
    examples/wanvideo/model_training/train.py \
    --dataset_base_path "data/tiger_dataset" \
    --dataset_metadata_path "$DATASET_PATH" \
    --dataset_repeat 25 \
    --num_epochs 20 \
    --model_id_with_origin_paths "${MODEL_PATHS}" \
    --lora_rank 32 \
    --learning_rate 1e-4 \
    --save_steps 1 \
    --output_path "$OUTPUT_PATH"

TRAIN_EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Fine-tuning completed successfully"
    echo ""
    echo "Output checkpoints:"
    ls -lh "$OUTPUT_PATH" | grep "epoch-"
    echo ""
    echo "Total checkpoints:"
    ls "$OUTPUT_PATH"/epoch-*.safetensors 2>/dev/null | wc -l
else
    echo "❌ Fine-tuning failed with exit code: $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi
echo "=================================================="
echo "Completed at: $(date)"
