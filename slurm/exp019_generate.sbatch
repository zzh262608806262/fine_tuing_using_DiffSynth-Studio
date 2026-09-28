#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH -t 3:00:00
#SBATCH -J exp019_gen
#SBATCH -o outputs/exp019_generate_%j.log
#SBATCH -e outputs/exp019_generate_%j.err

# Exp019: 视频生成
# 用法: sbatch --job-name=exp019_gen_graddiff_base slurm/exp019_generate.sbatch graddiff_base
#       sbatch --job-name=exp019_gen_graddiff_erased slurm/exp019_generate.sbatch graddiff_erased

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "用法: sbatch --job-name=exp019_gen_<arm> $0 <arm>"
    echo "可用臂: graddiff_base, graddiff_erased"
    exit 1
fi

ARM="$1"

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "========================================"
echo "Exp 019 视频生成: $ARM"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "========================================"

# 激活环境
source /home/x_jiage/jiage/video-unlearning/.venv/bin/activate
echo "✓ 已激活 VU .venv 环境"

# 设置环境变量
export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "[GENERATE] 开始生成视频..."
echo "  臂: $ARM"
echo "  数量: 99个视频"
echo "  输出: outputs/exp019/$ARM/"
echo "========================================"

python -u scripts/exp019_generate_videos.py \
    --arm "$ARM" \
    --start 0 \
    --end 99 \
    --frames 17 \
    --height 480 \
    --width 720 \
    --steps 50 \
    --cfg 5.0 \
    --flow-shift 5.0 \
    --fps 15

echo ""
echo "========================================"
echo "✓ 视频生成完成"
echo "输出目录: outputs/exp019/$ARM/"
echo "完成时间: $(date)"
echo "========================================"
