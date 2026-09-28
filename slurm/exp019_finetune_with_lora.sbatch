#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J exp019_ft_lora
#SBATCH -o outputs/exp019_finetune_lora_%j.log
#SBATCH -e outputs/exp019_finetune_lora_%j.err

set -euo pipefail

FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
cd "$FT_ROOT"

export PATH=~/mambaforge/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate "$FT_ROOT/.conda"

echo "=========================================="
echo "Exp019 Fine-tuning: Base + Unlearn LoRA"
echo "=========================================="
echo "Strategy: Load base model + apply unlearn LoRA, then train finetune LoRA"
echo "Base: Wan-AI/Wan2.2-TI2V-5B"
echo "Unlearn LoRA: models/unlearn/exp019_wan5b_nudity_graddiff/adapter_step600.pt"
echo "Output: models/finetune/exp019_graddiff_ft/"
echo ""

# 使用base模型（不用erased），但在训练时加载unlearn LoRA
python examples/wanvideo/model_training/train.py \
  --model_id_with_origin_paths Wan-AI/Wan2.2-TI2V-5B \
  --preset_lora_path models/unlearn/exp019_wan5b_nudity_graddiff/adapter_step600.pt \
  --dataset_base_path data/exp015 \
  --dataset_repeat 25 \
  --output_path models/finetune/exp019_graddiff_ft \
  --lora_rank 32 \
  --lora_target_modules qkv,mlp \
  --gradient_accumulation_steps 4 \
  --num_epochs 20 \
  --save_steps 500 \
  --learning_rate 0.0001 \
  --height 480 \
  --width 832 \
  --num_frames 17

echo "✓ Fine-tuning完成"
ls -lh models/finetune/exp019_graddiff_ft/
