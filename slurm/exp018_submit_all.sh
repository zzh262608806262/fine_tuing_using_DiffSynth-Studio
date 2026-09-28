#!/usr/bin/env bash
# ============================================================================
# Exp018 批量提交脚本 - 4 种擦除方法的高密度微调
#
# 方法: ESD, NPO, GradAscent, AnchorDistill
# 配置: repeat=25, epochs=20 → 20 个 checkpoints
# 基于: Exp015 的擦除模型
# ============================================================================

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

METHODS=("esd" "npo" "grad_ascent" "anchor_distill")

echo "=================================================="
echo "Exp018 批量提交 - 多方法高密度微调"
echo "=================================================="
echo "方法: ${METHODS[*]}"
echo "配置: repeat=25, epochs=20"
echo "预计时长: ~24-32 小时/方法"
echo "=================================================="
echo ""

# 验证擦除模型都存在
echo "🔍 前置检查:"
all_exist=true
for method in "${METHODS[@]}"; do
    model_dir="models/wan5b/exp015_${method}_erased"
    if [ -d "$model_dir" ]; then
        echo "  ✅ $model_dir"
    else
        echo "  ❌ $model_dir (不存在)"
        all_exist=false
    fi
done
echo ""

if [ "$all_exist" = false ]; then
    echo "❌ 错误: 部分擦除模型不存在，请先运行 Exp015 Stage 1-2"
    exit 1
fi

# 提交作业
echo "🚀 提交作业:"
declare -A job_ids

for method in "${METHODS[@]}"; do
    job_id=$(METHOD="$method" sbatch --job-name="exp018-ft-${method}" \
        slurm/exp018_finetune_multi_method.sbatch "$method" | awk '{print $4}')
    job_ids["$method"]=$job_id
    echo "  [$method] Job $job_id"
done

echo ""
echo "=================================================="
echo "提交完成"
echo "=================================================="
echo ""
echo "作业列表:"
for method in "${METHODS[@]}"; do
    echo "  ${method}: ${job_ids[$method]}"
done

echo ""
echo "日志位置:"
echo "  slurm/logs/exp018_finetune_<method>-<jobid>.out"
echo ""
echo "查看队列: squeue --me"
echo "取消全部: scancel ${job_ids[*]}"
echo ""
echo "产物位置:"
for method in "${METHODS[@]}"; do
    echo "  models/finetune/exp018_${method}_ft/epoch-{1..20}.safetensors"
done
echo ""
