#!/bin/bash
#SBATCH -A naiss2024-22-1588
#SBATCH -p berzelius
#SBATCH --gpus-per-node=1
#SBATCH -t 2:00:00
#SBATCH -J exp016_gen_nf4_{MODEL}
#SBATCH -o logs/slurm-%j.out

# Exp016 Stage 2: 生成NF4量化模型视频
# 模型: {MODEL}_nf4
# 输出: outputs/exp016/{MODEL}_nf4/

set -e

echo "========================================"
echo "Exp016 Stage 2: NF4视频生成"
echo "模型: {MODEL}_nf4"
echo "Job ID: $SLURM_JOB_ID"
echo "节点: $(hostname)"
echo "时间: $(date)"
echo "========================================"
echo

# 环境变量
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export HF_HUB_OFFLINE=1

# 激活环境
source /cephyr/users/jiage/Alvis/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

cd /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio

# 生成视频（全量99个）
python scripts/exp016_generate_nf4.py \
    --model {MODEL} \
    --start 0 \
    --end 99

echo
echo "✅ {MODEL}_nf4 生成完成"
echo "输出: outputs/exp016/{MODEL}_nf4/"
echo "完成时间: $(date)"
