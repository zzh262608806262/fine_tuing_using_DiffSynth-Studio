#!/bin/bash

# Exp015 Stage 5: 批量提交评估作业
# 8个臂并行评估

set -e

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

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

echo "========================================"
echo "Exp015 Stage 5: 批量提交评估作业"
echo "总计: 8个臂 × 99条 = 792个视频"
echo "========================================"
echo ""

JOB_IDS=()

for arm in "${ARMS[@]}"; do
    # 创建臂专用的 sbatch 文件
    sbatch_file="slurm/exp015_eval_${arm}.sbatch"
    sed "s/ARMNAME/${arm}/g" slurm/exp015_eval_template.sbatch > "$sbatch_file"

    echo "提交: $arm"
    job_id=$(sbatch "$sbatch_file" | awk '{print $4}')
    echo "  ✓ Job ID: $job_id"
    JOB_IDS+=($job_id)
done

echo ""
echo "========================================"
echo "已提交 ${#JOB_IDS[@]} 个作业:"
for job_id in "${JOB_IDS[@]}"; do
    echo "  - $job_id"
done
echo ""
echo "监控命令:"
echo "  squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
echo "========================================"
