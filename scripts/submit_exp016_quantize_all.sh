#!/bin/bash
# Exp016 Stage 1: 批量提交量化任务

set -e

MODELS=(
    "baseline"
    "esd_erased"
    "npo_erased"
    "grad_ascent_erased"
    "anchor_distill_erased"
)

TEMPLATE="slurm/exp016_quantize_template.sbatch"

echo "提交 Exp016 Stage 1 - 模型量化任务..."
echo ""

for model in "${MODELS[@]}"; do
    # 生成临时sbatch文件
    tmp_sbatch="/tmp/exp016_quant_${model}_$$.sbatch"
    sed "s/MODELNAME/${model}/g" "$TEMPLATE" > "$tmp_sbatch"

    # 提交
    echo "提交: $model"
    sbatch "$tmp_sbatch"

    # 清理
    rm "$tmp_sbatch"
done

echo ""
echo "✅ 已提交 ${#MODELS[@]} 个量化任务"
echo "查看状态: squeue -u \$USER | grep exp016-quant"
