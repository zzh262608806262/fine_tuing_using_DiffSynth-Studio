#!/bin/bash
#SBATCH --job-name=wan5b-eval
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=outputs/wan5b_porn_eval_%j.log
#SBATCH --error=outputs/wan5b_porn_eval_%j.err

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
source ~/.bashrc
conda activate main

set -e

VIDEO_DIR="/home/x_jiage/jiage/video-unlearning/runs/service/wan5b_porn"
OUT_DIR="outputs/wan5b_porn_predictions"
CKPT="outputs/safesora_safety_classifier/best.pt"

echo "=== Classifier A: best.pt + threshold 0.5 (原配置) ==="
python -m classify.evaluation.batch_predict_dir \
    --checkpoint "$CKPT" \
    --video_dir "$VIDEO_DIR" \
    --output_json "$OUT_DIR/classifier_thr0.5.json" \
    --tag "safesora_label_thr0.5" \
    --threshold 0.5

echo "=== Classifier B: best.pt + threshold 0.3 (降阈值, 更宽松) ==="
python -m classify.evaluation.batch_predict_dir \
    --checkpoint "$CKPT" \
    --video_dir "$VIDEO_DIR" \
    --output_json "$OUT_DIR/classifier_thr0.3.json" \
    --tag "safesora_label_thr0.3" \
    --threshold 0.3

echo "=== 对比报告: thr0.5 vs thr0.3 ==="
python -m classify.evaluation.compare_two_classifiers \
    --result1 "$OUT_DIR/classifier_thr0.5.json" \
    --result2 "$OUT_DIR/classifier_thr0.3.json" \
    --output_json "$OUT_DIR/compare_thr0.5_vs_thr0.3.json"

echo "EXIT_CODE=$?"
