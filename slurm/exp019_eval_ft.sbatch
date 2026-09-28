#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J exp019_eval_ft
#SBATCH -o outputs/exp019_eval_ft_%j.log
#SBATCH -e outputs/exp019_eval_ft_%j.err

module load Mambaforge/23.3.1-1-hpc1-bdist
eval "$(conda shell.bash hook)"
conda activate video-unlearning

mkdir -p outputs/exp019_evaluation

python scripts/evaluate_videos_nudenet.py \
    --video_dir outputs/exp019/graddiff_ft_e19 \
    --output_file outputs/exp019_evaluation/graddiff_ft_e19_evaluation.json \
    --threshold 0.6

echo "Exp019 微调后评估完成"
