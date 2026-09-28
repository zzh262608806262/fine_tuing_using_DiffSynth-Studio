#!/bin/bash
#SBATCH -A berzelius-2026-50
#SBATCH --gpus=1
#SBATCH -C "thin"
#SBATCH -t 2:00:00
#SBATCH -o /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio/slurm/logs/eval_porn_wan5b_quant_%j.log

# Wan2.2-TI2V-5B 量化模型安全评估

cd /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

echo "=========================================="
echo "量化模型安全评估"
echo "  ARMS: base_quant, erased_quant"
echo "  SET:  benchmark (99条)"
echo "=========================================="

python -u scripts/eval_porn_wan5b_quant.py \
    --arm base_quant,erased_quant \
    --set benchmark \
    --nudenet-off

echo ""
echo "评估完成，结果:"
ls -lh outputs/wan5b_eval_quant/classify_results/
