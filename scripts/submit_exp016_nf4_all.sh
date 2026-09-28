#!/bin/bash
# Exp016 Stage 2: 提交所有NF4模型视频生成任务

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TEMPLATE="slurm/exp016_gen_nf4_template.sbatch"
TMP_DIR="$CLAUDE_JOB_DIR/tmp"
mkdir -p "$TMP_DIR"
mkdir -p logs

# 5个量化模型
MODELS=(
    "baseline"
    "esd_erased"
    "npo_erased"
    "grad_ascent_erased"
    "anchor_distill_erased"
)

echo "========================================"
echo "Exp016 Stage 2: 提交NF4视频生成任务"
echo "时间: $(date)"
echo "========================================"
echo

JOB_IDS=()

for model in "${MODELS[@]}"; do
    echo "提交: ${model}_nf4"

    # 替换模板中的占位符
    sbatch_script="$TMP_DIR/exp016_gen_nf4_${model}.sbatch"
    sed "s/{MODEL}/$model/g" "$TEMPLATE" > "$sbatch_script"

    # 提交作业
    job_output=$(sbatch "$sbatch_script")
    job_id=$(echo "$job_output" | grep -oP '\d+')
    JOB_IDS+=("$job_id")

    echo "  ✓ Job ID: $job_id"
    echo
done

echo "========================================"
echo "提交完成"
echo "========================================"
echo
echo "作业列表："
for i in "${!MODELS[@]}"; do
    echo "  ${MODELS[$i]}_nf4: ${JOB_IDS[$i]}"
done
echo
echo "监控命令："
echo "  squeue -u \$USER"
echo "  watch -n 10 squeue -u \$USER"
echo
echo "预计产出："
echo "  5个模型 × 99个视频 = 495个视频"
echo "  输出目录: outputs/exp016/<model>_nf4/"
