#!/bin/bash
#SBATCH --job-name=st-cls
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/safesora_test_cls_%j.log
#SBATCH --error=outputs/safesora_test_cls_%j.err

# Exp 010: 分类器 best.pt 三阈值 on SafeSora test 集 5745 条
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
source ~/.bashrc
conda activate main

set -e

CKPT="outputs/safesora_safety_classifier/best.pt"
VIDEO_DIR="outputs/safesora_test_videos"
OUT_ROOT="outputs/safesora_test_multi_judge/test"

for thr in 0.50 0.30 0.20; do
  echo "=== [cls] threshold=$thr ==="
  python -u -m classify.evaluation.batch_predict_dir \
    --checkpoint "$CKPT" \
    --video_dir "$VIDEO_DIR" \
    --output_json "$OUT_ROOT/cls_thr_$thr.json" \
    --suffix "*.mp4" \
    --threshold "$thr" \
    --tag "safesora_test_cls_$thr"
done

echo "EXIT_CODE=$?"