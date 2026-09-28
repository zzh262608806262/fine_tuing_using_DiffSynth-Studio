#!/bin/bash
# Exp020a 一键提交脚本
# 依赖链：擦除训练 -> 合并 -> 生成 -> 评估

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "========================================"
echo "Exp020a 一键提交脚本"
echo "========================================"

# 提交擦除训练
echo "[1/4] 提交擦除训练..."
J_UNLEARN=$(sbatch slurm/exp020a_unlearn_npo_masked.sbatch | awk '{print $4}')
echo "  Job ID: $J_UNLEARN"

# 提交合并（依赖擦除训练）
echo "[2/4] 提交合并（依赖J=$J_UNLEARN）..."
J_MERGE=$(sbatch --dependency=afterok:$J_UNLEARN slurm/exp020a_merge.sbatch | awk '{print $4}')
echo "  Job ID: $J_MERGE"

# 提交生成（依赖合并）
echo "[3/4] 提交生成（依赖J=$J_MERGE）..."
J_GENERATE=$(sbatch --dependency=afterok:$J_MERGE slurm/exp020a_generate.sbatch | awk '{print $4}')
echo "  Job ID: $J_GENERATE"

# 提交评估（依赖生成）
echo "[4/4] 提交评估（依赖J=$J_GENERATE）..."
J_EVALUATE=$(sbatch --dependency=afterok:$J_GENERATE slurm/exp020a_evaluate.sbatch | awk '{print $4}')
echo "  Job ID: $J_EVALUATE"

echo ""
echo "========================================"
echo "✅ 全部作业已提交"
echo "========================================"
echo "Job依赖链:"
echo "  $J_UNLEARN (擦除训练)"
echo "    └─> $J_MERGE (合并)"
echo "          └─> $J_GENERATE (生成)"
echo "                └─> $J_EVALUATE (评估)"
echo ""
echo "监控命令:"
echo "  squeue -u x_jiage"
echo "  查看日志: ls -lht slurm/logs/exp020a_*"
echo "========================================"
