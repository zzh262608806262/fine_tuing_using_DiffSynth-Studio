#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 0:10:00
#SBATCH -J exp019_test_ft
#SBATCH -o outputs/exp019_test_finetune_%j.log
#SBATCH -e outputs/exp019_test_finetune_%j.err

# Exp019 微调命令测试（10分钟）

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "========================================"
echo "Exp019 微调命令测试"
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

echo ""
echo "Testing fine-tuning command..."
echo "========================================"

# 测试：1个repeat, 1个epoch，只训练1步就够了
timeout 300 accelerate launch --mixed_precision=bf16 --num_processes=1 \
    examples/wanvideo/model_training/train.py \
    --dataset_base_path "data/tiger_dataset" \
    --dataset_metadata_path "data/tiger_dataset/metadata_100.csv" \
    --dataset_repeat 1 \
    --num_epochs 1 \
    --model_id_with_origin_paths "wan5b/exp019_graddiff_erased:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.safetensors,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.safetensors" \
    --lora_rank 32 \
    --learning_rate 1e-4 \
    --save_steps 1 \
    --output_path "/tmp/test_exp019_finetune_${SLURM_JOB_ID}"

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
