#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J exp018_eval
#SBATCH -o slurm/logs/exp018_evaluate-%x-%j.out
#SBATCH -e slurm/logs/exp018_evaluate-%x-%j.err

# Exp018 Stage 5: 评估（NudeNet + 分类器）
# 用法: sbatch --job-name=exp018_eval_esd slurm/exp018_evaluate.sbatch esd

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "用法: sbatch --job-name=exp018_eval_<method> $0 <method>"
    echo "可用方法: esd, npo, grad_ascent, anchor_distill"
    exit 1
fi

METHOD="$1"
EPOCH="19"

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "========================================"
echo "Exp018 评估: ${METHOD} epoch-${EPOCH}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "========================================"

# 激活环境
source /home/x_jiage/jiage/video-unlearning/.venv/bin/activate
echo "✓ 已激活 VU .venv 环境"

export PYTHONNOUSERSITE=1

# 路径配置
VIDEO_DIR="outputs/exp018/${METHOD}_ft_e${EPOCH}"
OUTPUT_FILE="outputs/exp018/evaluation/${METHOD}_ft_e${EPOCH}_results.json"

echo ""
echo "[配置]"
echo "  方法: ${METHOD}"
echo "  视频: ${VIDEO_DIR}"
echo "  输出: ${OUTPUT_FILE}"
echo "========================================"

# 前置检查
if [ ! -d "$VIDEO_DIR" ]; then
    echo "❌ 视频目录不存在: $VIDEO_DIR"
    exit 1
fi

video_count=$(find "$VIDEO_DIR" -name "*.mp4" 2>/dev/null | wc -l)
if [ "$video_count" -eq 0 ]; then
    echo "❌ 视频目录为空: $VIDEO_DIR"
    exit 1
fi

echo "✅ 找到 ${video_count} 个视频"
echo ""

# 评估
echo "[EVALUATE] 开始评估..."
python -u scripts/exp018_evaluate.py \
    --method "$METHOD" \
    --video_dir "$VIDEO_DIR" \
    --output_file "$OUTPUT_FILE" \
    --nudenet_threshold 0.6

if [ $? -ne 0 ]; then
    echo "[ERROR] 评估失败"
    exit 1
fi

echo ""
echo "[DONE] 评估完成"
echo "  结果: $OUTPUT_FILE"

# 显示摘要
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "=== 结果摘要 ==="
    cat "$OUTPUT_FILE" | jq '.nudenet.summary'
fi

echo ""
echo "Time: $(date)"
echo "========================================"
