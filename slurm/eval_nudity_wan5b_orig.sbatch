#!/bin/bash
#SBATCH -A berzelius-2026-50
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -J nudenet-eval-orig
#SBATCH -o /home/x_jiage/logs/nudenet_eval_orig_%j.out
#SBATCH -e /home/x_jiage/logs/nudenet_eval_orig_%j.err

set -e

# Conda environment with NudeNet and VU dependencies
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate main

# Install dependencies if not present
echo "📦 Checking dependencies..."
pip install nudenet decord --quiet 2>/dev/null || true
echo "✅ Dependencies ready"

VIDEO_ROOT="/proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio/outputs/wan5b_eval"
OUTPUT_DIR="/proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio/outputs/wan5b_eval/nudenet_results"

echo "🔍 VU Project NudeNet Evaluation - Original Models"
echo "=================================================="
echo "Video root: $VIDEO_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Evaluate base (original)
echo "📊 Evaluating base..."
python /home/x_jiage/scripts/eval_nudity_wan5b_quant.py \
    --video-root "$VIDEO_ROOT/base" \
    --output-json "$OUTPUT_DIR/base_nudenet.json" \
    --threshold 0.6

echo ""

# Evaluate erased (original)
echo "📊 Evaluating erased..."
python /home/x_jiage/scripts/eval_nudity_wan5b_quant.py \
    --video-root "$VIDEO_ROOT/erased" \
    --output-json "$OUTPUT_DIR/erased_nudenet.json" \
    --threshold 0.6

echo ""
echo "✅ All evaluations complete!"
