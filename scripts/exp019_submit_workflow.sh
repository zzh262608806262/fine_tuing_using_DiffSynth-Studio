#!/bin/bash
# Exp019完整工作流提交脚本
# GradDiff补充实验：对比擦除前后的安全性

set -euo pipefail

cd "$(dirname "$(dirname "$(readlink -f "$0")")")" # Auto: project root

echo "========================================"
echo "Exp019 完整工作流提交"
echo "实验: GradDiff擦除效果验证"
echo "时间: $(date)"
echo "========================================"

# Phase 1: 擦除训练（已完成 Job 17530967）
echo ""
echo "[Phase 1] 擦除训练 - 已完成"
echo "  Job ID: 17530967"
echo "  输出: models/unlearn/exp019_wan5b_nudity_graddiff/"
echo "  状态: ✓ 完成（7个checkpoint）"

# Phase 2: LoRA合并
echo ""
echo "[Phase 2] LoRA合并"

# 2.1 Base模型（创建软链接）
J_MERGE_BASE=$(sbatch --parsable slurm/exp019_merge_base.sbatch)
echo "  2.1 Base模型准备: Job $J_MERGE_BASE"

# 2.2 Erased模型（合并adapter）
J_MERGE_ERASED=$(sbatch --parsable slurm/exp019_merge_erased.sbatch)
echo "  2.2 Erased合并: Job $J_MERGE_ERASED"

# Phase 3: 视频生成（等待merge完成）
echo ""
echo "[Phase 3] 视频生成（99条nudity prompts）"

J_GEN_BASE=$(sbatch --parsable \
    --dependency=afterok:$J_MERGE_BASE \
    --job-name=exp019_gen_graddiff_base \
    slurm/exp019_generate.sbatch graddiff_base)
echo "  3.1 Base生成: Job $J_GEN_BASE (依赖 $J_MERGE_BASE)"

J_GEN_ERASED=$(sbatch --parsable \
    --dependency=afterok:$J_MERGE_ERASED \
    --job-name=exp019_gen_graddiff_erased \
    slurm/exp019_generate.sbatch graddiff_erased)
echo "  3.2 Erased生成: Job $J_GEN_ERASED (依赖 $J_MERGE_ERASED)"

# Phase 4: NudeNet评估（等待生成完成）
echo ""
echo "[Phase 4] NudeNet评估"

J_EVAL_BASE=$(sbatch --parsable \
    --dependency=afterok:$J_GEN_BASE \
    --job-name=exp019_eval_graddiff_base \
    slurm/exp019_eval.sbatch graddiff_base)
echo "  4.1 Base评估: Job $J_EVAL_BASE (依赖 $J_GEN_BASE)"

J_EVAL_ERASED=$(sbatch --parsable \
    --dependency=afterok:$J_GEN_ERASED \
    --job-name=exp019_eval_graddiff_erased \
    slurm/exp019_eval.sbatch graddiff_erased)
echo "  4.2 Erased评估: Job $J_EVAL_ERASED (依赖 $J_GEN_ERASED)"

# 保存Job ID记录
JOB_RECORD="outputs/exp019/job_ids.txt"
mkdir -p outputs/exp019
cat > "$JOB_RECORD" <<EOF
Exp019 Job IDs
提交时间: $(date)

Phase 1: 擦除训练（已完成）
  Unlearn: 17530967

Phase 2: LoRA合并
  Base: $J_MERGE_BASE
  Erased: $J_MERGE_ERASED

Phase 3: 视频生成
  Base: $J_GEN_BASE
  Erased: $J_GEN_ERASED

Phase 4: NudeNet评估
  Base: $J_EVAL_BASE
  Erased: $J_EVAL_ERASED
EOF

echo ""
echo "========================================"
echo "✓ 所有作业已提交"
echo ""
echo "作业链:"
echo "  Phase 2: $J_MERGE_BASE (base), $J_MERGE_ERASED (erased)"
echo "  Phase 3: $J_GEN_BASE ← $J_MERGE_BASE"
echo "           $J_GEN_ERASED ← $J_MERGE_ERASED"
echo "  Phase 4: $J_EVAL_BASE ← $J_GEN_BASE"
echo "           $J_EVAL_ERASED ← $J_GEN_ERASED"
echo ""
echo "Job ID记录: $JOB_RECORD"
echo ""
echo "监控命令:"
echo "  squeue -u \$USER"
echo "  tail -f outputs/exp019_*.log"
echo "========================================"
