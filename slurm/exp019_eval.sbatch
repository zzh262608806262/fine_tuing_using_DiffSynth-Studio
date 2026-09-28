#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J exp019_eval
#SBATCH -o outputs/exp019_eval_%j.log
#SBATCH -e outputs/exp019_eval_%j.err

# Exp019: NudeNet评估
# 用法: sbatch --job-name=exp019_eval_graddiff_base slurm/exp019_eval.sbatch graddiff_base
#       sbatch --job-name=exp019_eval_graddiff_erased slurm/exp019_eval.sbatch graddiff_erased

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "用法: sbatch --job-name=exp019_eval_<arm> $0 <arm>"
    echo "可用臂: graddiff_base, graddiff_erased"
    exit 1
fi

ARM="$1"

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "========================================"
echo "Exp 019 NudeNet评估: $ARM"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "========================================"

# 激活环境
VU_ROOT="/home/x_jiage/jiage/video-unlearning"
source "$VU_ROOT/.venv/bin/activate"
echo "✓ 已激活 VU .venv 环境"

VIDEO_DIR="outputs/exp019/$ARM"
OUTPUT_DIR="outputs/exp019/eval"

if [ ! -d "$VIDEO_DIR" ]; then
    echo "[ERROR] 视频目录不存在: $VIDEO_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "[EVAL] 开始NudeNet评估..."
echo "  输入: $VIDEO_DIR"
echo "  方法: $ARM"
echo "  输出: $OUTPUT_DIR/${ARM}.jsonl"
echo "========================================"

python "$VU_ROOT/scripts/eval/score_nudity_rate.py" \
    --video-root "$VIDEO_DIR" \
    --method "$ARM" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "✓ NudeNet评估完成"
echo "输出目录: $OUTPUT_DIR"

# 打印统计
OUTPUT_FILE="$OUTPUT_DIR/${ARM}.jsonl"
if [ -f "$OUTPUT_FILE" ]; then
    NUM_VIDEOS=$(wc -l < "$OUTPUT_FILE")
    NUM_UNSAFE=$(grep '"is_unsafe": true' "$OUTPUT_FILE" | wc -l)
    UNSAFE_RATE=$(python -c "print(f'{$NUM_UNSAFE/$NUM_VIDEOS*100:.2f}%')")

    echo ""
    echo "统计 ($ARM):"
    echo "  视频数: $NUM_VIDEOS"
    echo "  unsafe数: $NUM_UNSAFE"
    echo "  unsafe率: $UNSAFE_RATE"
fi

echo "完成时间: $(date)"
echo "========================================"
