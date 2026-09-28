#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -J exp019_gen_distill
#SBATCH -o outputs/exp019_generate_distill_%j.log
#SBATCH -e outputs/exp019_generate_distill_%j.err

# Exp019 Phase 14: 蒸馏后视频生成（4步推理）
# 用法: sbatch --export=EPOCH=5 slurm/exp019_generate_distilled.sbatch

set -euo pipefail

if [ -z "${EPOCH:-}" ]; then
    echo "❌ Error: EPOCH environment variable not set"
    exit 1
fi

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "========================================"
echo "Exp019 Phase 14 - 蒸馏后生成（4步）"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Epoch: $EPOCH"
echo "========================================"

# 激活环境
source /home/x_jiage/jiage/video-unlearning/.venv/bin/activate
export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 检查蒸馏模型
MODEL_DIR="models/train/exp019_graddiff_distill"
MODEL_FILE="$MODEL_DIR/epoch-${EPOCH}.safetensors"

if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Error: Distilled model not found: $MODEL_FILE"
    exit 1
fi

echo ""
echo "[GENERATE] 蒸馏模型生成（4步推理）"
echo "  模型: $MODEL_FILE"
echo "  输出: outputs/exp019/graddiff_distilled_epoch${EPOCH}/"
echo "========================================"

# 使用基线生成脚本，但修改steps为4
python -u scripts/exp019_generate_videos.py \
    --arm graddiff_distilled_epoch${EPOCH} \
    --start 0 \
    --end 99 \
    --frames 17 \
    --height 480 \
    --width 720 \
    --steps 4 \
    --cfg 5.0 \
    --flow-shift 5.0 \
    --fps 15

echo ""
echo "========================================"
echo "✓ 蒸馏视频生成完成（4步）"
echo "完成时间: $(date)"
echo "========================================"
