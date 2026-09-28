#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 4:00:00
#SBATCH -J exp016-gen-ARMNAME
#SBATCH -o slurm/logs/exp016_gen_ARMNAME_%j.out
#SBATCH -e slurm/logs/exp016_gen_ARMNAME_%j.err

# Exp016: 量化模型视频生成
# 模板变量: ARMNAME

set -e

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

echo "=================================================="
echo "Exp016: 生成 ARMNAME"
echo "=================================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo "=================================================="
echo ""
echo "生成配置:"
echo "  臂: ARMNAME"
echo "  提示词数: 99"
echo "  输出: outputs/exp016/ARMNAME"
echo ""

cd /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio

python -u scripts/exp016_generate_videos.py \
    --arm ARMNAME \
    --start 0 \
    --end 99 \
    --output-dir "outputs/exp016/ARMNAME"

echo ""
echo "=================================================="
echo "✅ ARMNAME 生成完成"
echo "完成时间: $(date)"
echo "=================================================="
