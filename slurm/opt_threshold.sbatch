#!/bin/bash
#SBATCH --job-name=opt-thresh
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/outputs/opt_threshold_%j.log
#SBATCH --error=/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/outputs/opt_threshold_%j.err

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
source ~/.bashrc
conda activate main

python -m classify.evaluation.optimize_threshold \
    --checkpoint outputs/safesora_safety_classifier/best.pt \
    --test_annotation /home/x_jiage/jiage/datasets/SafeSora-Label/test.jsonl \
    --video_root /home/x_jiage/jiage/datasets/SafeSora \
    --output_json outputs/safesora_safety_classifier/binary_threshold.json

echo "EXIT_CODE=$?"
