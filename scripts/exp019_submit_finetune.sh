#!/bin/bash
# Exp019 Phase 5-7 提交脚本：微调 → 生成 → 评估

set -euo pipefail

cd "$(dirname "$(dirname "$(readlink -f "$0")")")" # Auto: project root

echo "========================================"
echo "Exp019 Phase 5-7 提交：微调实验"
echo "时间: $(date)"
echo "========================================"

# Phase 5: 微调训练（24小时）
echo ""
echo "[Phase 5] 微调训练"
J_FINETUNE=$(sbatch --parsable slurm/exp019_finetune.sbatch)
echo "  微调训练: Job $J_FINETUNE"

# Phase 6-7: 选择关键checkpoint生成和评估
# 选择: epoch-1, 5, 10, 15, 20（5个点绘制回潮曲线）
EPOCHS=(1 5 10 15 20)

echo ""
echo "[Phase 6-7] 微调后生成与评估（5个checkpoint）"

declare -a GEN_JOBS
declare -a EVAL_JOBS

for EPOCH in "${EPOCHS[@]}"; do
    # 生成作业（依赖微调完成）
    J_GEN=$(sbatch --parsable \
        --dependency=afterok:$J_FINETUNE \
        --export=EPOCH=$EPOCH \
        slurm/exp019_generate_finetuned.sbatch)
    GEN_JOBS+=($J_GEN)

    # 评估作业（依赖生成完成）
    J_EVAL=$(sbatch --parsable \
        --dependency=afterok:$J_GEN \
        --export=EPOCH=$EPOCH \
        slurm/exp019_eval_finetuned.sbatch)
    EVAL_JOBS+=($J_EVAL)

    echo "  Epoch $EPOCH: Gen=$J_GEN, Eval=$J_EVAL"
done

# 保存Job ID记录
JOB_RECORD="outputs/exp019/job_ids_finetune.txt"
cat >> "$JOB_RECORD" <<EOF

================================================
Exp019 Phase 5-7: 微调实验
提交时间: $(date)

Phase 5: 微调训练
  Finetune: $J_FINETUNE

Phase 6-7: 生成与评估（关键checkpoint）
EOF

for i in "${!EPOCHS[@]}"; do
    echo "  Epoch ${EPOCHS[$i]}: Gen=${GEN_JOBS[$i]}, Eval=${EVAL_JOBS[$i]}" >> "$JOB_RECORD"
done

echo "" >> "$JOB_RECORD"
echo "================================================" >> "$JOB_RECORD"

echo ""
echo "========================================"
echo "✓ 所有作业已提交"
echo ""
echo "作业链:"
echo "  Phase 5: $J_FINETUNE (微调24h)"
for i in "${!EPOCHS[@]}"; do
    echo "  Epoch ${EPOCHS[$i]}: ${GEN_JOBS[$i]} ← $J_FINETUNE → ${EVAL_JOBS[$i]}"
done
echo ""
echo "Job ID记录: $JOB_RECORD"
echo ""
echo "监控命令:"
echo "  squeue -u \$USER"
echo "  tail -f outputs/exp019_finetune_*.log"
echo "========================================"
