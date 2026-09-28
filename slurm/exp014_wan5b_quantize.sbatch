#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 8:00:00
#SBATCH -J exp014-quantize
#SBATCH -o logs/exp014_quantize_%j.out
#SBATCH -e logs/exp014_quantize_%j.err

# Exp 014: NF4量化4个臂（每臂20个epoch checkpoint）

set -e

echo "=================================================="
echo "Exp 014 - NF4 Quantization (4 arms × 20 epochs)"
echo "=================================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo ""

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

echo "📊 量化任务:"
echo "  - 臂1: base (无擦除，无微调)"
echo "  - 臂2: erased (擦除，无微调)"
echo "  - 臂3: base_ft (无擦除，微调20个epoch)"
echo "  - 臂4: erased_ft (擦除，微调20个epoch)"
echo ""

# 臂1: base（仅1个模型）
echo "🔧 [1/4] 量化base模型..."
python scripts/quantize_wan5b.py \
    --model-path "models/wan5b/exp014_base" \
    --output-dir "models/quantized/exp014_base_nf4" \
    --quantization-method nf4

echo "✅ base量化完成"
echo ""

# 臂2: erased（仅1个模型）
echo "🔧 [2/4] 量化erased模型..."
python scripts/quantize_wan5b.py \
    --model-path "models/wan5b/exp014_erased" \
    --output-dir "models/quantized/exp014_erased_nf4" \
    --quantization-method nf4

echo "✅ erased量化完成"
echo ""

# 臂3: base_ft（20个epoch checkpoint）
echo "🔧 [3/4] 量化base_ft模型（20个epoch）..."
for epoch in $(seq -f "%02g" 1 20); do
    echo "  处理 epoch_${epoch}..."
    python scripts/quantize_wan5b.py \
        --model-path "models/finetune/exp014_base_ft/epoch_${epoch}" \
        --lora-path "models/finetune/exp014_base_ft/epoch_${epoch}/lora" \
        --output-dir "models/quantized/exp014_base_ft_nf4/epoch_${epoch}" \
        --quantization-method nf4
done

echo "✅ base_ft量化完成（20个epoch）"
echo ""

# 臂4: erased_ft（20个epoch checkpoint）
echo "🔧 [4/4] 量化erased_ft模型（20个epoch）..."
for epoch in $(seq -f "%02g" 1 20); do
    echo "  处理 epoch_${epoch}..."
    python scripts/quantize_wan5b.py \
        --model-path "models/finetune/exp014_erased_ft/epoch_${epoch}" \
        --lora-path "models/finetune/exp014_erased_ft/epoch_${epoch}/lora" \
        --output-dir "models/quantized/exp014_erased_ft_nf4/epoch_${epoch}" \
        --quantization-method nf4
done

echo "✅ erased_ft量化完成（20个epoch）"
echo ""

echo "=================================================="
echo "✅ Exp 014量化阶段完成！"
echo "完成时间: $(date)"
echo ""
echo "产物统计："
echo "  - base: 1个模型"
echo "  - erased: 1个模型"
echo "  - base_ft: 20个模型"
echo "  - erased_ft: 20个模型"
echo "  总计: 42个量化模型"
echo ""
echo "下一步: 运行评估脚本"
echo "  sbatch slurm/exp014_wan5b_evaluate.sbatch"
echo "=================================================="
