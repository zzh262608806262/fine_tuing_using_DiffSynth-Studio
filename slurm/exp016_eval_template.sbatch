#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 2:00:00
#SBATCH -J exp016-eval-ARMNAME
#SBATCH -o slurm/logs/exp016_eval_ARMNAME_%j.out
#SBATCH -e slurm/logs/exp016_eval_ARMNAME_%j.err

# Exp016: 量化模型评估
# 模板变量: ARMNAME

set -e

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

echo "=================================================="
echo "Exp016: 评估 ARMNAME"
echo "=================================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "=================================================="
echo ""
echo "评估配置:"
echo "  臂: ARMNAME"
echo "  视频目录: outputs/exp016/ARMNAME"
echo "  输出: outputs/exp016_evaluation/ARMNAME_evaluation.json"
echo ""

cd /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio

python -u scripts/exp016_evaluate.py \
    --arm ARMNAME \
    --output-dir outputs/exp016_evaluation

echo ""
echo "=================================================="
echo "✅ ARMNAME 评估完成"
echo "完成时间: $(date)"
echo "=================================================="
