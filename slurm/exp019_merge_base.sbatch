#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 0:05:00
#SBATCH -J exp019_merge_base
#SBATCH -o outputs/exp019_merge_base_%j.log
#SBATCH -e outputs/exp019_merge_base_%j.err

# Exp019: base臂不需要merge，直接用原始Wan2.2-TI2V-5B
# 这个脚本只是创建软链接标记

set -euo pipefail

FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
cd "$FT_ROOT"

echo "Exp019 base臂: 使用原始Wan2.2-TI2V-5B（无需merge）"
echo "创建软链接: models/wan5b/exp019_graddiff_base -> models/Wan-AI/Wan2.2-TI2V-5B"

ln -sfn "$FT_ROOT/models/Wan-AI/Wan2.2-TI2V-5B" "$FT_ROOT/models/wan5b/exp019_graddiff_base"

echo "✓ base臂准备完成"
ls -lh models/wan5b/exp019_graddiff_base
