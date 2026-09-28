#!/bin/bash

# Exp017 蒸馏视频评估 - 提交脚本

echo "========================================"
echo "Exp017 蒸馏评估任务提交"
echo "时间: $(date)"
echo "========================================"

# 检查冒烟测试是否成功
if [ ! -f "outputs/exp017_evaluation/base_distill_evaluation.json" ]; then
    echo "⚠️  建议先运行冒烟测试验证脚本正确性"
    echo "   sbatch slurm/exp017_eval_smoke.sbatch"
    echo ""
    read -p "是否继续提交完整任务？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi
fi

# 提交评估作业（5个方法并行）
JOB_ID=$(sbatch slurm/exp017_eval_distill.sbatch | awk '{print $4}')

echo ""
echo "✅ 已提交 5 个评估任务（array job）"
echo "Job ID: ${JOB_ID}"
echo "环境: vu (包含 nudenet)"
echo ""
echo "监控命令:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f outputs/exp017_eval_distill_${JOB_ID}_*.log"
echo ""
echo "预计时间: ~30分钟/方法（5个并行，总计~30分钟）"
echo ""
echo "完成后查看结果:"
echo "  ls -lh outputs/exp017_evaluation/"
echo ""
