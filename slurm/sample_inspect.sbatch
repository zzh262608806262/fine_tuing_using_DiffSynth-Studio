#!/bin/bash
#SBATCH --job-name=inspect
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=outputs/sample_inspect_%j.log
#SBATCH --error=outputs/sample_inspect_%j.err

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
source ~/.bashrc
conda activate main

python -m classify.evaluation.sample_inspect \
    --checkpoint outputs/safesora_safety_classifier/best.pt \
    --test_annotation /home/x_jiage/jiage/datasets/SafeSora-Label/test.jsonl \
    --video_root /home/x_jiage/jiage/datasets/SafeSora \
    --n 40 \
    --output_dir outputs/inspect_sample

echo "EXIT_CODE=$?"
