#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J exp019_ft_base
#SBATCH -o outputs/exp019_finetune_base_%j.log
#SBATCH -e outputs/exp019_finetune_base_%j.err

set -euo pipefail

FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
cd "$FT_ROOT"

export PATH=~/mambaforge/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate "$FT_ROOT/.conda"

echo "=========================================="
echo "Exp019 Fine-tuning on Base Model"
echo "=========================================="
echo "Strategy: Train on base model (not erased)"
echo "Base: Wan-AI/Wan2.2-TI2V-5B"
echo "Dataset: data/exp015 (I2P nudity prompts)"
echo "Output: models/finetune/exp019_base_ft/"
echo ""

python examples/wanvideo/model_training/train.py \
  --model_id_with_origin_paths Wan-AI/Wan2.2-TI2V-5B \
  --dataset_base_path data/exp015 \
  --dataset_repeat 25 \
  --output_path models/finetune/exp019_base_ft \
  --lora_rank 32 \
  --lora_target_modules qkv,mlp \
  --gradient_accumulation_steps 4 \
  --num_epochs 20 \
  --save_steps 500 \
  --learning_rate 0.0001 \
  --height 480 \
  --width 832 \
  --num_frames 17

echo ""
echo "✓ Fine-tuning完成"
ls -lh models/finetune/exp019_base_ft/
