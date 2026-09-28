#!/bin/bash
#SBATCH -A berzelius-2026-50
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH -t 4:00:00
#SBATCH -o /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio/slurm/logs/wan5b_eval_quant_%A_${ARM}_${SET}.log

# Wan2.2-TI2V-5B 量化模型评测视频生成
# 用法: sbatch --export=ARM=erased_quant,SET=fast,BEGIN=0,END=29 slurm/wan5b_eval_quant_gen.sbatch

cd /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

export DIFFSYNTH_SKIP_DOWNLOAD=true

# 默认参数
ARM=${ARM:-erased_quant}
SET=${SET:-fast}
BEGIN=${BEGIN:-0}
END=${END:-29}

echo "=========================================="
echo "Wan2.2-TI2V-5B 量化评测生成"
echo "  ARM:   $ARM"
echo "  SET:   $SET"
echo "  RANGE: $BEGIN..$((END-1))"
echo "=========================================="

mkdir -p outputs/wan5b_eval_quant

python -u scripts/generate_wan5b_eval_quant_v2.py \
    --arm "$ARM" \
    --set "$SET" \
    --start "$BEGIN" \
    --end "$END"

echo "任务完成: ARM=$ARM, SET=$SET, RANGE=$BEGIN..$((END-1))"
