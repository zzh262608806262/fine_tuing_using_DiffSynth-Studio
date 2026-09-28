#!/bin/bash
# Exp016: 批量提交补充生成任务

ARMS=(
    "esd_erased_nf4"
    "esd_base_nf4"
    "npo_erased_nf4"
    "npo_base_nf4"
    "grad_ascent_erased_nf4"
    "grad_ascent_base_nf4"
    "anchor_distill_erased_nf4"
    "anchor_distill_base_nf4"
)

TEMPLATE="slurm/exp016_regen_template.sbatch"

echo "提交 Exp016 补充生成任务..."
echo ""

for arm in "${ARMS[@]}"; do
    echo "提交: $arm"
    sed "s/ARMNAME/$arm/g" "$TEMPLATE" | sbatch
    sleep 1
done

echo ""
echo "✅ 已提交 ${#ARMS[@]} 个补充生成任务"
echo "查看状态: squeue -u \$USER | grep exp016-regen"
