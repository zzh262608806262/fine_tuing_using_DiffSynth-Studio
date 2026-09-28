#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 3-00:00:00
#SBATCH -J exp019_ft_graddiff
#SBATCH -o outputs/exp019_finetune_graddiff_%j.log
#SBATCH -e outputs/exp019_finetune_graddiff_%j.err

set -euo pipefail

FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
cd "$FT_ROOT"

export PATH=~/mambaforge/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate /home/x_jiage/.conda/envs/main

export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Exp019 GradDiff Fine-tuning"
echo "=========================================="
echo "Model: exp019_graddiff_erased"
echo "Dataset: data/tiger_dataset (100 videos)"
echo "Repeat: 25, Epochs: 20"
echo "Output: models/finetune/exp019_graddiff_ft/"
echo ""

# 构造model_id_with_origin_paths（逗号分隔）
MODEL_ID="wan5b/exp019_graddiff_erased"
MODEL_PATHS="${MODEL_ID}:diffusion_pytorch_model*.safetensors,${MODEL_ID}:models_t5_umt5-xxl-enc-bf16.safetensors,${MODEL_ID}:Wan2.2_VAE.safetensors"

python examples/wanvideo/model_training/train.py \
  --model_id_with_origin_paths "$MODEL_PATHS" \
  --dataset_base_path data/tiger_dataset \
  --dataset_metadata_path data/tiger_dataset/metadata_100.csv \
  --dataset_repeat 25 \
  --output_path models/finetune/exp019_graddiff_ft \
  --num_epochs 20 \
  --save_steps 500 \
  --lora_base_model dit \
  --lora_rank 32 \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 0.0001 \
  --height 480 \
  --width 832 \
  --num_frames 17

echo ""
echo "✓ Fine-tuning完成"
ls -lh models/finetune/exp019_graddiff_ft/
