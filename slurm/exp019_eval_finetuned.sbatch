#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J exp019_eval_ft
#SBATCH -o outputs/exp019_eval_ft_%j.log
#SBATCH -e outputs/exp019_eval_ft_%j.err

# Exp019 Phase 7: 微调后评估
# 用法: sbatch --export=EPOCH=1 slurm/exp019_eval_finetuned.sbatch

set -euo pipefail

if [ -z "${EPOCH:-}" ]; then
    echo "❌ Error: EPOCH environment variable not set"
    echo "Usage: sbatch --export=EPOCH=1 slurm/exp019_eval_finetuned.sbatch"
    exit 1
fi

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "========================================"
echo "Exp019 Phase 7 - 微调后评估"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Epoch: $EPOCH"
echo "Time: $(date)"
echo "========================================"

# 激活环境
VU_ROOT="/home/x_jiage/jiage/video-unlearning"
source "$VU_ROOT/.venv/bin/activate"
echo "✓ 已激活 VU .venv 环境"

VIDEO_DIR="outputs/exp019/graddiff_finetuned_epoch${EPOCH}"
OUTPUT_FILE="outputs/exp019/graddiff_finetuned_epoch${EPOCH}_nudenet.jsonl"

if [ ! -d "$VIDEO_DIR" ]; then
    echo "❌ Error: 视频目录不存在: $VIDEO_DIR"
    exit 1
fi

echo ""
echo "[EVAL] 开始NudeNet评估..."
echo "  输入: $VIDEO_DIR"
echo "  输出: $OUTPUT_FILE"
echo "========================================"

python "$VU_ROOT/scripts/eval/score_nudity_rate.py" \
    --video-root "$VIDEO_DIR" \
    --output "$OUTPUT_FILE" \
    --device cuda

echo ""
echo "========================================"
echo "✓ NudeNet评估完成"
echo "输出文件: $OUTPUT_FILE"

# 打印统计
if [ -f "$OUTPUT_FILE" ]; then
    NUM_VIDEOS=$(wc -l < "$OUTPUT_FILE")
    NUM_UNSAFE=$(grep '"is_unsafe": true' "$OUTPUT_FILE" | wc -l)
    UNSAFE_RATE=$(python -c "print(f'{$NUM_UNSAFE/$NUM_VIDEOS*100:.2f}%')")

    echo ""
    echo "统计 (Epoch $EPOCH):"
    echo "  视频数: $NUM_VIDEOS"
    echo "  unsafe数: $NUM_UNSAFE"
    echo "  unsafe率: $UNSAFE_RATE"
fi

echo "完成时间: $(date)"
echo "========================================"
