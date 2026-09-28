#!/bin/bash
# Exp019 Phase 9-12 量化实验提交脚本

set -euo pipefail

cd "$(dirname "$(dirname "$(readlink -f "$0")")")" # Auto: project root

echo "========================================"
echo "Exp019 Phase 9-12: 量化实验"
echo "时间: $(date)"
echo "========================================"

# Phase 9: 量化转换
echo ""
echo "[Phase 9] 量化转换"
J_QUANTIZE=$(sbatch --parsable slurm/exp019_quantize.sbatch)
echo "  量化: Job $J_QUANTIZE"

# Phase 10: 量化生成（依赖量化完成）
echo ""
echo "[Phase 10] 量化模型生成"
J_GEN_QUANT=$(sbatch --parsable \
    --dependency=afterok:$J_QUANTIZE \
    --job-name=exp019_gen_quantized \
    --export=MODEL_DIR=models/wan5b/exp019_graddiff_erased_int8,OUTPUT_DIR=outputs/exp019/graddiff_quantized \
    slurm/exp019_generate.sbatch graddiff_quantized)
echo "  生成: Job $J_GEN_QUANT"

# Phase 11: 量化评估
echo ""
echo "[Phase 11] 量化评估"
J_EVAL_QUANT=$(sbatch --parsable \
    --dependency=afterok:$J_GEN_QUANT \
    slurm/exp019_eval.sbatch graddiff_quantized)
echo "  评估: Job $J_EVAL_QUANT"

# Phase 12: 量化微调（依赖量化完成）
echo ""
echo "[Phase 12] 量化微调"
J_FT_QUANT=$(sbatch --parsable \
    --dependency=afterok:$J_QUANTIZE \
    --export=ERASED_MODEL=models/wan5b/exp019_graddiff_erased_int8,OUTPUT_PATH=models/finetune/exp019_graddiff_int8_lora \
    slurm/exp019_finetune.sbatch)
echo "  微调: Job $J_FT_QUANT"

# 保存Job ID
JOB_RECORD="outputs/exp019/job_ids_quantize.txt"
cat > "$JOB_RECORD" <<EOF
================================================
Exp019 Phase 9-12: 量化实验
提交时间: $(date)

Phase 9: 量化转换
  Quantize: $J_QUANTIZE

Phase 10: 量化生成
  Generate: $J_GEN_QUANT

Phase 11: 量化评估
  Evaluate: $J_EVAL_QUANT

Phase 12: 量化微调
  Finetune: $J_FT_QUANT
================================================
EOF

echo ""
echo "========================================"
echo "✓ 量化实验作业已提交"
echo ""
echo "作业链:"
echo "  Phase 9: $J_QUANTIZE (量化)"
echo "  Phase 10: $J_GEN_QUANT ← $J_QUANTIZE"
echo "  Phase 11: $J_EVAL_QUANT ← $J_GEN_QUANT"
echo "  Phase 12: $J_FT_QUANT ← $J_QUANTIZE"
echo ""
echo "Job ID记录: $JOB_RECORD"
echo "========================================"
