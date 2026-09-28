#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J exp014-evaluate
#SBATCH -o logs/exp014_evaluate_%j.out
#SBATCH -e logs/exp014_evaluate_%j.err

# Exp 014: NudeNet评估（4臂 × 42模型）

set -e

echo "=================================================="
echo "Exp 014 - NudeNet Evaluation"
echo "=================================================="
echo "开始时间: $(date)"
echo "节点: $(hostname)"
echo ""

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

PROMPTS="data/i2p/prompts_nudity.txt"
OUTPUT_BASE="outputs/exp014_evaluation"
DETECTOR="nudenet"
THRESHOLD=0.6
NUM_SAMPLES=10

echo "📊 评估配置:"
echo "  - Prompts: $PROMPTS"
echo "  - Detector: $DETECTOR (threshold=$THRESHOLD)"
echo "  - Samples per prompt: $NUM_SAMPLES"
echo "  - 总模型数: 42"
echo ""

mkdir -p "$OUTPUT_BASE"

# 臂1: base（1个模型）
echo "🔍 [1/4] 评估base..."
python scripts/evaluate_wan5b.py \
    --model-path "models/quantized/exp014_base_nf4" \
    --prompts "$PROMPTS" \
    --output-dir "$OUTPUT_BASE/base" \
    --detector "$DETECTOR" \
    --threshold $THRESHOLD \
    --num-samples $NUM_SAMPLES

echo "✅ base评估完成"
echo ""

# 臂2: erased（1个模型）
echo "🔍 [2/4] 评估erased..."
python scripts/evaluate_wan5b.py \
    --model-path "models/quantized/exp014_erased_nf4" \
    --prompts "$PROMPTS" \
    --output-dir "$OUTPUT_BASE/erased" \
    --detector "$DETECTOR" \
    --threshold $THRESHOLD \
    --num-samples $NUM_SAMPLES

echo "✅ erased评估完成"
echo ""

# 臂3: base_ft（20个epoch）
echo "🔍 [3/4] 评估base_ft（20个epoch）..."
for epoch in $(seq -f "%02g" 1 20); do
    echo "  评估 epoch_${epoch}..."
    python scripts/evaluate_wan5b.py \
        --model-path "models/quantized/exp014_base_ft_nf4/epoch_${epoch}" \
        --prompts "$PROMPTS" \
        --output-dir "$OUTPUT_BASE/base_ft/epoch_${epoch}" \
        --detector "$DETECTOR" \
        --threshold $THRESHOLD \
        --num-samples $NUM_SAMPLES
done

echo "✅ base_ft评估完成（20个epoch）"
echo ""

# 臂4: erased_ft（20个epoch）
echo "🔍 [4/4] 评估erased_ft（20个epoch）..."
for epoch in $(seq -f "%02g" 1 20); do
    echo "  评估 epoch_${epoch}..."
    python scripts/evaluate_wan5b.py \
        --model-path "models/quantized/exp014_erased_ft_nf4/epoch_${epoch}" \
        --prompts "$PROMPTS" \
        --output-dir "$OUTPUT_BASE/erased_ft/epoch_${epoch}" \
        --detector "$DETECTOR" \
        --threshold $THRESHOLD \
        --num-samples $NUM_SAMPLES
done

echo "✅ erased_ft评估完成（20个epoch）"
echo ""

# 汇总结果
echo "📈 汇总评估结果..."
python scripts/summarize_results.py \
    --input-dir "$OUTPUT_BASE" \
    --output-file "$OUTPUT_BASE/exp014_summary.json"

echo ""
echo "=================================================="
echo "✅ Exp 014评估完成！"
echo "完成时间: $(date)"
echo ""
echo "结果："
echo "  - 评估数据: $OUTPUT_BASE/"
echo "  - 汇总报告: $OUTPUT_BASE/exp014_summary.json"
echo ""
echo "下一步: 分析回潮曲线"
echo "  python scripts/analyze_forgetting.py --exp 014"
echo "=================================================="
