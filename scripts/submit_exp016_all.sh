#!/bin/bash
# Exp016: 批量提交所有8个臂的量化任务

ARMS=(
    "esd_erased"
    "esd_base"
    "npo_erased"
    "npo_base"
    "grad_ascent_erased"
    "grad_ascent_base"
    "anchor_distill_erased"
    "anchor_distill_base"
)

TEMPLATE="slurm/exp016_quant_template.sbatch"

for arm in "${ARMS[@]}"; do
    echo "提交: $arm"
    sed "s/ARMNAME/$arm/g" "$TEMPLATE" | sbatch
    sleep 1
done

echo ""
echo "✅ 已提交 ${#ARMS[@]} 个量化任务"
echo "查看状态: squeue -u \$USER"
