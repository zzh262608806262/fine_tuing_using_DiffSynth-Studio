#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 00:30:00
#SBATCH -J exp020a_smoke
#SBATCH -o slurm/logs/exp020a_smoke-%j.out
#SBATCH -e slurm/logs/exp020a_smoke-%j.err

set -e

echo "=========================================="
echo "NPOMasked 完整训练 Smoke Test (1 step)"
echo "=========================================="

cd /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio

# 激活VU环境（包含所有依赖）
VU_ROOT="/home/x_jiage/jiage/video-unlearning"
source "$VU_ROOT/.venv/bin/activate"

echo "环境: $(which python)"
echo "开始时间: $(date)"
echo ""

# 运行1步训练
python scripts/run_unlearn_wan5b.py \
  --train \
  --method NPOMasked \
  --erase-concept "breasts" \
  --model-path models/Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --forget-manifest data/splits/nudity_forget_composed_wan.jsonl \
  --train-dir models/unlearned/smoke_test_npo_masked \
  --unlearn-steps 1 \
  --save-every 999999

echo ""
echo "完成时间: $(date)"
echo "=========================================="
echo "✅ Smoke test 成功！可以提交完整训练"
echo "=========================================="
