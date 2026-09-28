#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=A100-SXM4-80GB:1
#SBATCH -t 3-00:00:00
#SBATCH -J exp019_distill
#SBATCH -o outputs/exp019_distill_%j.log
#SBATCH -e outputs/exp019_distill_%j.err

# Exp019 Phase 13: GradDiff擦除模型蒸馏训练（50步→4步）

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "========================================"
echo "Exp019 Phase 13 - 蒸馏训练"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "========================================"

# 激活环境
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

# 路径配置
TEACHER_DIT="models/wan5b/exp019_graddiff_erased"
BASE_MODEL="models/Wan-AI/Wan2.2-TI2V-5B"
OUTPUT_PATH="models/train/exp019_graddiff_distill"
DATASET_PATH="data/tiger_dataset/metadata_100_distill.csv"

echo "Configuration:"
echo "  Teacher DiT: $TEACHER_DIT"
echo "  Base (T5/VAE): $BASE_MODEL"
echo "  Output: $OUTPUT_PATH"
echo "  Dataset: $DATASET_PATH"
echo "  Distill steps: 50 -> 4"
echo "  Epochs: 10"
echo "========================================"
echo ""

# 前置检查
echo "🔍 Checking prerequisites..."

if [ ! -d "$TEACHER_DIT" ]; then
    echo "❌ Error: Teacher DiT not found: $TEACHER_DIT"
    exit 1
fi

if [ ! -d "$BASE_MODEL" ]; then
    echo "❌ Error: Base model not found: $BASE_MODEL"
    exit 1
fi

# 检查DiT文件
dit_files=(
    "diffusion_pytorch_model-00001-of-00003.safetensors"
    "diffusion_pytorch_model-00002-of-00003.safetensors"
    "diffusion_pytorch_model-00003-of-00003.safetensors"
)

for file in "${dit_files[@]}"; do
    if [ ! -f "$TEACHER_DIT/$file" ]; then
        echo "❌ Error: Missing DiT file: $TEACHER_DIT/$file"
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

# 构建模型路径列表（嵌套数组格式，参考 Exp017）
# DiT: 3个分片作为内层数组
DIT_PATHS="["
for file in "${dit_files[@]}"; do
    DIT_PATHS+="\"${TEACHER_DIT}/${file}\","
done
DIT_PATHS="${DIT_PATHS%,}]"

# 外层数组: [DiT数组, T5, VAE]
MODEL_PATHS="[${DIT_PATHS},"
for file in "${auxiliary_files[@]}"; do
    MODEL_PATHS+="\"${BASE_MODEL}/${file}\","
done
MODEL_PATHS="${MODEL_PATHS%,}]"

echo "Model paths structure:"
echo "$MODEL_PATHS"
echo ""

echo "🚀 Starting distillation training..."
echo "=================================================="

# 使用 distill_wan5b.py 包装脚本（Exp017验证过的）
python scripts/distill_wan5b.py \
    --mode train \
    --no_download_dataset \
    --dataset_base_path "data/tiger_dataset" \
    --dataset_metadata_path "$DATASET_PATH" \
    --output_path "$OUTPUT_PATH" \
    --dataset_repeat 10 \
    --num_epochs 10 \
    --learning_rate 1e-5 \
    --height 480 \
    --width 736 \
    --num_frames 17 \
    --model_paths "$MODEL_PATHS" \
    --extra_inputs seed,rand_device \
    --use_gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --accelerate_config examples/wanvideo/model_training/full/accelerate_config_single_gpu.yaml

TRAIN_EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Distillation training completed successfully"
    echo ""
    echo "Output checkpoints:"
    ls -lh "$OUTPUT_PATH" | grep "epoch-"
    echo ""
    echo "Total checkpoints:"
    ls "$OUTPUT_PATH"/epoch-*.safetensors 2>/dev/null | wc -l
else
    echo "❌ Distillation training failed with exit code: $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi
echo "=================================================="
echo "Completed at: $(date)"
