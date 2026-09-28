#!/usr/bin/env bash
# ============================================================================
# Exp018 Stage 4 批量提交 - 视频生成（带 dependency）
#
# 自动等待 Exp018 Stage 3 的微调任务完成后启动生成
# ============================================================================

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# Exp018 Stage 3 的 Job IDs（微调任务）
METHODS=("esd" "npo" "grad_ascent" "anchor_distill")
FT_JOBS=(17493434 17493435 17493436 17493437)

echo "=================================================="
echo "Exp018 Stage 4 批量提交 - 视频生成"
echo "=================================================="
echo "策略: 使用 --dependency=afterok 自动等待微调完成"
echo ""

echo "🔍 检查微调任务状态:"
for i in "${!METHODS[@]}"; do
    method="${METHODS[$i]}"
    job_id="${FT_JOBS[$i]}"

    # 检查任务状态
    status=$(sacct -j "$job_id" --format=State --noheader | head -1 | tr -d ' ')

    echo "  [$method] Job $job_id: $status"

    if [ "$status" = "COMPLETED" ]; then
        echo "    ✅ 已完成，dependency 会立即满足"
    elif [ "$status" = "RUNNING" ]; then
        echo "    🔄 运行中，将等待完成"
    elif [ "$status" = "PENDING" ]; then
        echo "    ⏳ 排队中，将等待完成"
    else
        echo "    ⚠️  状态异常: $status"
    fi
done

echo ""
echo "🚀 提交生成任务（带 dependency）:"

declare -A gen_jobs

for i in "${!METHODS[@]}"; do
    method="${METHODS[$i]}"
    ft_job="${FT_JOBS[$i]}"

    # 提交任务，设置 dependency
    gen_job=$(sbatch \
        --dependency=afterok:${ft_job} \
        --job-name="exp018_gen_${method}" \
        slurm/exp018_generate.sbatch "$method" | awk '{print $4}')

    gen_jobs["$method"]=$gen_job
    echo "  [$method] Gen Job $gen_job (depends on FT $ft_job)"
done

echo ""
echo "=================================================="
echo "提交完成"
echo "=================================================="
echo ""
echo "生成任务列表:"
for method in "${METHODS[@]}"; do
    echo "  ${method}: ${gen_jobs[$method]}"
done

echo ""
echo "日志位置:"
echo "  slurm/logs/exp018_generate_exp018_gen_<method>-<jobid>.out"
echo ""
echo "产物位置:"
for method in "${METHODS[@]}"; do
    echo "  outputs/exp018/${method}_ft_e19/*.mp4 (99个视频)"
done
echo ""
echo "查看队列: squeue --me"
echo "查看依赖: squeue --me -o '%.10i %.20j %.8T %.20E'"
echo ""
