#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH -t 2:00:00
#SBATCH -J exp019_gen_ft
#SBATCH -o outputs/exp019_generate_ft_%j.log
#SBATCH -e outputs/exp019_generate_ft_%j.err

# Exp019 Phase 6: 微调后视频生成
# 用法: sbatch --export=EPOCH=1 slurm/exp019_generate_finetuned.sbatch
#       sbatch --export=EPOCH=20 slurm/exp019_generate_finetuned.sbatch

set -euo pipefail

if [ -z "${EPOCH:-}" ]; then
    echo "❌ Error: EPOCH environment variable not set"
    echo "Usage: sbatch --export=EPOCH=1 slurm/exp019_generate_finetuned.sbatch"
    exit 1
fi

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "========================================"
echo "Exp019 Phase 6 - 微调后视频生成"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Epoch: $EPOCH"
echo "Time: $(date)"
echo "========================================"

# 激活环境
source /home/x_jiage/jiage/video-unlearning/.venv/bin/activate
echo "✓ 已激活 VU .venv 环境"

# 设置环境变量
export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 检查LoRA文件
LORA_PATH="models/finetune/exp019_graddiff_erased_lora/epoch-${EPOCH}.safetensors"
if [ ! -f "$LORA_PATH" ]; then
    echo "❌ Error: LoRA checkpoint not found: $LORA_PATH"
    exit 1
fi

echo ""
echo "[GENERATE] 开始生成视频..."
echo "  基座: models/wan5b/exp019_graddiff_erased"
echo "  LoRA: $LORA_PATH"
echo "  输出: outputs/exp019/graddiff_finetuned_epoch${EPOCH}/"
echo "========================================"

python -u scripts/exp019_generate_finetuned.py \
    --epoch "$EPOCH" \
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
echo "输出目录: outputs/exp019/graddiff_finetuned_epoch${EPOCH}/"
echo "完成时间: $(date)"
echo "========================================"
