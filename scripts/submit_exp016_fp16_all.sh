#!/bin/bash
# Exp016 Stage 0: 批量提交FP16基线生成任务

set -e

MODELS=(
    "baseline"
    "esd_erased"
    "npo_erased"
    "grad_ascent_erased"
    "anchor_distill_erased"
)

TEMPLATE="slurm/exp016_gen_fp16_template.sbatch"

echo "提交 Exp016 Stage 0 - FP16基线生成任务..."
echo ""

for model in "${MODELS[@]}"; do
    # 生成临时sbatch文件
    tmp_sbatch="/tmp/exp016_fp16_${model}_$$.sbatch"
    sed "s/MODELNAME/${model}/g" "$TEMPLATE" > "$tmp_sbatch"

    # 提交
    echo "提交: $model"
    sbatch "$tmp_sbatch"

    # 清理
    rm "$tmp_sbatch"
done

echo ""
echo "✅ 已提交 ${#MODELS[@]} 个FP16基线生成任务"
echo "查看状态: squeue -u \$USER | grep exp016-fp16"
