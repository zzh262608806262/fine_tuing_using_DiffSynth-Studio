#!/usr/bin/env bash
# Exp014 测试流程提交脚本
#
# 用法: bash slurm/exp014_submit_test.sh
#
# 说明：
#   - Stage 1: 擦除训练（600步，约2小时）
#   - Stage 2: LoRA合并 + 桥转换（约30分钟，依赖Stage 1）
#   - 使用 --dependency=afterok 确保顺序执行
#
# 注意：这是测试版本，只跑前两个阶段

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "=================================================="
echo "Exp014 测试流程提交"
echo "=================================================="
echo "时间: $(date)"
echo ""

# ---- Stage 1: 擦除训练（600步） ----
echo "提交 Stage 1: 擦除训练（600步，VU标准配置）"
J1=$(sbatch --parsable slurm/exp014_stage1_test.sbatch)
echo "  Job ID: $J1"
echo ""

# ---- Stage 2: LoRA合并（依赖Stage 1） ----
echo "提交 Stage 2: LoRA合并 + 桥转换（依赖 Stage 1）"
J2=$(sbatch --parsable --dependency=afterok:$J1 slurm/exp014_stage2_test.sbatch)
echo "  Job ID: $J2"
echo ""

# ---- 保存记录 ----
mkdir -p slurm/logs
LOG="slurm/logs/exp014_test_jobs.txt"
{
    echo "Exp014 测试作业 - $(date)"
    echo "================================"
    echo "Stage 1 (擦除训练): $J1"
    echo "Stage 2 (LoRA合并): $J2"
    echo ""
    echo "监控命令:"
    echo "  squeue --me"
    echo "  watch -n 60 'squeue --me'"
} > "$LOG"

echo "=================================================="
echo "✅ 提交完成！"
echo "=================================================="
echo ""
echo "作业信息已保存到: $LOG"
echo ""
echo "监控命令:"
echo "  squeue --me"
echo "  watch -n 60 'squeue --me'"
echo ""
echo "查看日志:"
echo "  tail -f slurm/logs/exp014_stage1_test-${J1}.out"
echo "  tail -f slurm/logs/exp014_stage2_test-${J2}.out"
echo ""
echo "预计完成时间:"
echo "  Stage 1: 约 2 小时"
echo "  Stage 2: 约 30 分钟"
echo "  总计: 约 2.5 小时"
echo ""
