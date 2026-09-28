#!/usr/bin/env bash
# ============================================================================
# Exp018 Stage 5 批量提交 - 评估（带 dependency）
#
# 自动等待 Exp018 Stage 4 的视频生成完成后启动评估
# ============================================================================

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# Exp018 Stage 4 的 Job IDs（生成任务）
METHODS=("esd" "npo" "grad_ascent" "anchor_distill")
GEN_JOBS=(17501159 17501160 17501161 17501162)

echo "=================================================="
echo "Exp018 Stage 5 批量提交 - 评估"
echo "=================================================="
echo "策略: 使用 --dependency=afterok 自动等待生成完成"
echo ""

echo "🔍 检查生成任务状态:"
for i in "${!METHODS[@]}"; do
    method="${METHODS[$i]}"
    job_id="${GEN_JOBS[$i]}"

    # 检查任务状态
    status=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null || echo "NOT_FOUND")

    if [ "$status" = "NOT_FOUND" ]; then
        # 检查历史状态
        status=$(sacct -j "$job_id" --format=State --noheader | head -1 | tr -d ' ' || echo "UNKNOWN")
    fi

    echo "  [$method] Gen Job $job_id: $status"

    if [ "$status" = "COMPLETED" ]; then
        echo "    ✅ 已完成，dependency 会立即满足"
    elif [ "$status" = "RUNNING" ]; then
        echo "    🔄 运行中，将等待完成"
    elif [ "$status" = "PENDING" ]; then
        echo "    ⏳ 排队中，将等待完成"
    else
        echo "    ⚠️  状态: $status"
    fi
done

echo ""
echo "🚀 提交评估任务（带 dependency）:"

declare -A eval_jobs

for i in "${!METHODS[@]}"; do
    method="${METHODS[$i]}"
    gen_job="${GEN_JOBS[$i]}"

    # 提交任务，设置 dependency
    eval_job=$(sbatch \
        --dependency=afterok:${gen_job} \
        --job-name="exp018_eval_${method}" \
        slurm/exp018_evaluate.sbatch "$method" | awk '{print $4}')

    eval_jobs["$method"]=$eval_job
    echo "  [$method] Eval Job $eval_job (depends on Gen $gen_job)"
done

echo ""
echo "=================================================="
echo "提交完成"
echo "=================================================="
echo ""
echo "评估任务列表:"
for method in "${METHODS[@]}"; do
    echo "  ${method}: ${eval_jobs[$method]}"
done

echo ""
echo "日志位置:"
echo "  slurm/logs/exp018_evaluate_exp018_eval_<method>-<jobid>.out"
echo ""
echo "结果位置:"
for method in "${METHODS[@]}"; do
    echo "  outputs/exp018/evaluation/${method}_ft_e19_results.json"
done
echo ""
echo "查看队列: squeue --me"
echo "查看依赖: squeue --me -o '%.10i %.20j %.8T %.20E'"
echo ""
