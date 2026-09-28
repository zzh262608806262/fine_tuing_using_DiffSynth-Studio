#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J exp019_eval_baseline
#SBATCH --array=0-1
#SBATCH -o outputs/exp019_eval_baseline_%A_%a.log
#SBATCH -e outputs/exp019_eval_baseline_%A_%a.err

module load Mambaforge/24.1.2-0
eval "$(conda shell.bash hook)"
conda activate vu

ARMS=(graddiff_base graddiff_erased)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}

mkdir -p outputs/exp019_evaluation

python scripts/evaluate_videos_nudenet.py \
    --video_dir outputs/exp019/${ARM} \
    --output_file outputs/exp019_evaluation/${ARM}_evaluation.json \
    --threshold 0.6

echo "Exp019 ${ARM} 评估完成"
