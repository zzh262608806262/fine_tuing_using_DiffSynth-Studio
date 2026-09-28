#!/bin/bash
#SBATCH -A berzelius-2026-50
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J nudenet-eval-quant
#SBATCH -o /home/x_jiage/logs/nudenet_eval_quant_%j.out
#SBATCH -e /home/x_jiage/logs/nudenet_eval_quant_%j.err

set -e

# Conda environment with NudeNet and VU dependencies
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

# Install dependencies if not present
echo "📦 Checking dependencies..."
pip install nudenet decord --quiet 2>/dev/null || true
echo "✅ Dependencies ready"

VIDEO_ROOT="/proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio/outputs/wan5b_eval_quant"
OUTPUT_DIR="/proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio/outputs/wan5b_eval_quant/nudenet_results"

echo "🔍 VU Project NudeNet Evaluation - Quantized Models"
echo "=================================================="
echo "Video root: $VIDEO_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Evaluate base_quant
echo "📊 Evaluating base_quant..."
python /home/x_jiage/scripts/eval_nudity_wan5b_quant.py \
    --video-root "$VIDEO_ROOT/base_quant" \
    --output-json "$OUTPUT_DIR/base_quant_nudenet.json" \
    --threshold 0.6

echo ""

# Evaluate erased_quant
echo "📊 Evaluating erased_quant..."
python /home/x_jiage/scripts/eval_nudity_wan5b_quant.py \
    --video-root "$VIDEO_ROOT/erased_quant" \
    --output-json "$OUTPUT_DIR/erased_quant_nudenet.json" \
    --threshold 0.6

echo ""
echo "✅ All evaluations complete!"
