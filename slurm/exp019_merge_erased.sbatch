#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 0:15:00
#SBATCH -J exp019_merge_erased
#SBATCH -o outputs/exp019_merge_erased_%j.log
#SBATCH -e outputs/exp019_merge_erased_%j.err

set -euo pipefail

FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
VU_ROOT="/home/x_jiage/jiage/video-unlearning"
cd "$FT_ROOT"

# 激活 VU 环境
if [ -f "$VU_ROOT/.venv/bin/activate" ]; then
    source "$VU_ROOT/.venv/bin/activate"
else
    echo "[ERROR] 未找到 VU .venv 环境"
    exit 1
fi

echo "Exp019 erased merge: GradDiff adapter + Wan2.2-TI2V-5B"
echo "Adapter: models/unlearn/exp019_wan5b_nudity_graddiff/adapter_step600.pt"
echo "Output: models/wan5b/exp019_graddiff_erased"

# Erased 模型（合并 LoRA adapter）
python scripts/merge_unlearn_lora_wan5b.py \
    --adapter models/unlearn/exp019_wan5b_nudity_graddiff/adapter_step600.pt \
    --base-dir models/Wan-AI/Wan2.2-TI2V-5B-Diffusers/transformer \
    --output-dir models/wan5b/exp019_graddiff_erased_diffusers \
    --bridge-output-dir models/wan5b/exp019_graddiff_erased \
    --force

echo "✓ erased merge完成"
ls -lh models/wan5b/exp019_graddiff_erased/
