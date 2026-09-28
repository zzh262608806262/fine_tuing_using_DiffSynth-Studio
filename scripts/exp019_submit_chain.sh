#!/bin/bash
# Exp019 自动化提交脚本
# 用法: bash scripts/exp019_submit_chain.sh

set -e

echo "=== Exp019 GradDiff 补充实验 自动化提交 ==="
echo "开始时间: $(date)"
echo ""

# Stage 1: 擦除训练 (已手动提交: Job 17530936)
echo "Stage 1: 擦除训练 - 已提交 Job 17530936"
UNLEARN_JOB=17530936

# Stage 2: LoRA 合并 (依赖 Stage 1)
echo "Stage 2: 提交 LoRA 合并作业 (依赖 Job ${UNLEARN_JOB})"
MERGE_BASE=$(sbatch --dependency=afterok:${UNLEARN_JOB} slurm/exp019_merge_base.sbatch | awk '{print $4}')
MERGE_ERASED=$(sbatch --dependency=afterok:${UNLEARN_JOB} slurm/exp019_merge_erased.sbatch | awk '{print $4}')
echo "  - Base: Job ${MERGE_BASE}"
echo "  - Erased: Job ${MERGE_ERASED}"

# Stage 3: 基线生成 (依赖 Stage 2)
echo "Stage 3: 提交基线生成作业 (依赖 Jobs ${MERGE_BASE},${MERGE_ERASED})"
GEN_BASELINE=$(sbatch --dependency=afterok:${MERGE_BASE}:${MERGE_ERASED} slurm/exp019_generate_baseline.sbatch | awk '{print $4}')
echo "  - 生成: Job ${GEN_BASELINE}"

# Stage 4: 基线评估 (依赖 Stage 3)
echo "Stage 4: 提交基线评估作业 (依赖 Job ${GEN_BASELINE})"
EVAL_BASELINE=$(sbatch --dependency=afterok:${GEN_BASELINE} slurm/exp019_eval_baseline.sbatch | awk '{print $4}')
echo "  - 评估: Job ${EVAL_BASELINE}"

# Stage 5: 微调训练 (依赖 Stage 2)
echo "Stage 5: 提交微调训练作业 (依赖 Job ${MERGE_ERASED})"
FINETUNE=$(sbatch --dependency=afterok:${MERGE_ERASED} slurm/exp019_finetune.sbatch | awk '{print $4}')
echo "  - 微调: Job ${FINETUNE}"

# Stage 6: 微调后生成 (依赖 Stage 5)
echo "Stage 6: 提交微调后生成作业 (依赖 Job ${FINETUNE})"
GEN_FT=$(sbatch --dependency=afterok:${FINETUNE} slurm/exp019_generate_ft.sbatch | awk '{print $4}')
echo "  - 生成: Job ${GEN_FT}"

# Stage 7: 微调后评估 (依赖 Stage 6)
echo "Stage 7: 提交微调后评估作业 (依赖 Job ${GEN_FT})"
EVAL_FT=$(sbatch --dependency=afterok:${GEN_FT} slurm/exp019_eval_ft.sbatch | awk '{print $4}')
echo "  - 评估: Job ${EVAL_FT}"

echo ""
echo "=== 所有作业已提交 ==="
echo "完成时间: $(date)"
echo ""
echo "作业链："
echo "  擦除: ${UNLEARN_JOB}"
echo "  合并: ${MERGE_BASE}, ${MERGE_ERASED}"
echo "  生成: ${GEN_BASELINE}, ${GEN_FT}"
echo "  评估: ${EVAL_BASELINE}, ${EVAL_FT}"
echo "  微调: ${FINETUNE}"
echo ""
echo "监控命令:"
echo "  squeue -u \$USER"
echo "  tail -f outputs/exp019_*.log"
