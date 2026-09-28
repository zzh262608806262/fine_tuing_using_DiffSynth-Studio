#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 0:10:00
#SBATCH -J exp019_test_ft2
#SBATCH -o outputs/exp019_test_finetune2_%j.log
#SBATCH -e outputs/exp019_test_finetune2_%j.err

# Exp019 微调命令测试v2 - 使用model_paths JSON格式

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "========================================"
echo "Exp019 微调命令测试 v2 (JSON)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "========================================"

# 激活环境
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

export LD_LIBRARY_PATH="/home/x_jiage/.conda/envs/diffsynth/lib:$LD_LIBRARY_PATH"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "Environment:"
python --version
nvidia-smi --query-gpu=name --format=csv,noheader

# 构建绝对路径
BASE_DIR="/proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio"
ERASED_DIT="$BASE_DIR/models/wan5b/exp019_graddiff_erased"
BASE_MODEL="$BASE_DIR/models/Wan-AI/Wan2.2-TI2V-5B"

# 构建JSON格式的model_paths
MODEL_PATHS_JSON='[
    "'$ERASED_DIT'/diffusion_pytorch_model-00001-of-00003.safetensors",
    "'$ERASED_DIT'/diffusion_pytorch_model-00002-of-00003.safetensors",
    "'$ERASED_DIT'/diffusion_pytorch_model-00003-of-00003.safetensors",
    "'$BASE_MODEL'/models_t5_umt5-xxl-enc-bf16.safetensors",
    "'$BASE_MODEL'/Wan2.2_VAE.safetensors"
]'

echo ""
echo "Model paths JSON:"
echo "$MODEL_PATHS_JSON"
echo ""
echo "Testing fine-tuning command..."
echo "========================================"

# 测试：1个repeat, 1个epoch
timeout 300 accelerate launch --mixed_precision=bf16 --num_processes=1 \
    examples/wanvideo/model_training/train.py \
    --dataset_base_path "data/tiger_dataset" \
    --dataset_metadata_path "data/tiger_dataset/metadata_100.csv" \
    --dataset_repeat 1 \
    --num_epochs 1 \
    --model_paths "$MODEL_PATHS_JSON" \
    --lora_rank 32 \
    --learning_rate 1e-4 \
    --save_steps 1 \
    --output_path "/tmp/test_exp019_finetune2_${SLURM_JOB_ID}"

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 124 ]; then
    echo "✓ 测试成功：训练正常启动（5分钟后超时，符合预期）"
    exit 0
elif [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 测试成功：训练完成"
    exit 0
else
    echo "❌ 测试失败：退出码 $EXIT_CODE"
    exit $EXIT_CODE
fi
