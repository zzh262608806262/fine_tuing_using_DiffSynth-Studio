#!/bin/bash
# Exp019 Phase 13-16 蒸馏实验提交脚本（修复版）

set -euo pipefail

cd "$(dirname "$(dirname "$(readlink -f "$0")")")" # Auto: project root

echo "========================================"
echo "Exp019 Phase 13-16: 蒸馏实验（修复版）"
echo "时间: $(date)"
echo "========================================"
echo ""
echo "🔧 修复内容:"
echo "  - 使用 distill_wan5b.py 包装脚本（Exp017验证）"
echo "  - 修复模型路径格式（嵌套数组）"
echo "  - 解决 Job 17536004 的 ValueError"
echo ""

# Phase 13: 蒸馏训练（50步→4步）
echo "[Phase 13] 蒸馏训练"
J_DISTILL=$(sbatch --parsable slurm/exp019_distill_train.sbatch)
echo "  蒸馏训练: Job $J_DISTILL (48小时)"

# Phase 14-15: 蒸馏生成与评估（只生成最后一个checkpoint: epoch-10）
EPOCH=10

echo ""
echo "[Phase 14-15] 蒸馏后生成与评估（epoch-10）"

# 生成作业
J_GEN=$(sbatch --parsable \
    --dependency=afterok:$J_DISTILL \
    --export=EPOCH=$EPOCH,MODEL_DIR=models/train/exp019_graddiff_distill \
    slurm/exp019_generate_distilled.sbatch)

# 评估作业
J_EVAL=$(sbatch --parsable \
    --dependency=afterok:$J_GEN \
    slurm/exp019_eval.sbatch graddiff_distilled_epoch${EPOCH})

echo "  Epoch $EPOCH: Gen=$J_GEN, Eval=$J_EVAL"

# 保存Job ID
JOB_RECORD="outputs/exp019/job_ids_distill.txt"
cat > "$JOB_RECORD" <<EOF
================================================
Exp019 Phase 13-16: 蒸馏实验
提交时间: $(date)

Phase 13: 蒸馏训练
  Distill: $J_DISTILL

Phase 14-15: 生成与评估
EOF

for i in "${!EPOCHS[@]}"; do
    echo "  Epoch ${EPOCHS[$i]}: Gen=${GEN_JOBS[$i]}, Eval=${EVAL_JOBS[$i]}" >> "$JOB_RECORD"
done

echo "" >> "$JOB_RECORD"
echo "================================================" >> "$JOB_RECORD"

echo ""
echo "========================================"
echo "✓ 蒸馏实验作业已提交"
echo ""
echo "作业链:"
echo "  Phase 13: $J_DISTILL (蒸馏48h)"
for i in "${!EPOCHS[@]}"; do
    echo "  Epoch ${EPOCHS[$i]}: ${GEN_JOBS[$i]} ← $J_DISTILL → ${EVAL_JOBS[$i]}"
done
echo ""
echo "Job ID记录: $JOB_RECORD"
echo "========================================"
