#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-50
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH -t 3-00:00:00
#SBATCH -J lora100-rebuild
#SBATCH -o slurm/logs/lora100-%j.out
#SBATCH -e slurm/logs/lora100-%j.err

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

# 第一步: 用 Qwen3-VL-8B 为 100 个片段生成 caption, 重建 metadata_100.csv
if [ ! -f data/tiger_dataset/metadata_100.csv ] || \
   [ "$(wc -l < data/tiger_dataset/metadata_100.csv)" -lt 101 ]; then
    python scripts/caption_tiger_clips.py \
        --clip_dir data/tiger_dataset \
        --output_csv data/tiger_dataset/metadata_100.csv \
        --num_clips 100 --seed 0
fi

# 第二步: 重训 lora_100 (参数与原 training_args.json 对齐:
# repeat=100, epochs=5, lr=1e-4, rank=32, 480x832, num_frames=17)
python scripts/lora_finetune.py \
    --no_download_dataset \
    --dataset_base_path data/tiger_dataset \
    --dataset_metadata_path data/tiger_dataset/metadata_100.csv \
    --dataset_repeat 100 \
    --num_epochs 5 \
    --num_frames 17 \
    --height 480 --width 832 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --save_steps 2000 \
    --output_path ./models/train/Wan2.1-T2V-1.3B_lora_100 \
    --enable_tensorboard_log

echo "DONE lora100 rebuild"
ls -la ./models/train/Wan2.1-T2V-1.3B_lora_100/
