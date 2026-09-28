#!/usr/bin/env bash
# Exp015 Stage 4: 批量提交8个臂的视频生成作业

set -euo pipefail

cd "$(dirname "$0")/.."

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
echo "Exp015 Stage 4: 批量提交视频生成作业"
echo "总计: 8个臂 × 99条 = 792个视频"
echo "========================================"
echo ""

JOB_IDS=()

for arm in "${ARMS[@]}"; do
    job_name="exp015_gen_${arm}"

    echo "提交: $arm"
    job_id=$(sbatch --job-name="$job_name" slurm/exp015_generate.sbatch "$arm" | awk '{print $NF}')

    if [ -n "$job_id" ]; then
        echo "  ✓ Job ID: $job_id"
        JOB_IDS+=("$job_id")
    else
        echo "  ✗ 提交失败"
    fi
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
