#!/bin/bash
#SBATCH --job-name=smoke_test_mask
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/smoke_test_mask-%j.out
#SBATCH --error=slurm/logs/smoke_test_mask-%j.err

# Exp020a前置：验证Wan模型的mask参数
# 用法: sbatch slurm/smoke_test_wan_mask.sbatch

set -euo pipefail

echo "========================================"
echo "Wan Mask参数验证 - Smoke Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "========================================"

# 项目根目录
FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
cd "$FT_ROOT"

# 激活环境（使用VU环境，有完整的diffusers）
echo "[ENV] 激活环境..."
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"

VU_ROOT="/home/x_jiage/jiage/video-unlearning"
if [ -f "$VU_ROOT/.venv/bin/activate" ]; then
    echo "  使用VU .venv环境"
    source "$VU_ROOT/.venv/bin/activate"
elif conda env list | grep -q "video-unlearning"; then
    echo "  使用conda video-unlearning环境"
    conda activate video-unlearning
else
    echo "[ERROR] 未找到VU环境"
    exit 1
fi

# 环境验证
echo "[ENV] 验证依赖..."
python -c "import torch; print(f'  torch: {torch.__version__}')"
python -c "import diffusers; print(f'  diffusers: {diffusers.__version__}')"

# 模型路径
MODEL_PATH="$FT_ROOT/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"

echo ""
echo "[CONFIG] 配置:"
echo "  模型: $MODEL_PATH"
echo "  视频尺寸: 17帧 × 480h × 736w"

# 运行smoke test
echo ""
echo "[TEST] 运行smoke test..."
echo "========================================"

python3 scripts/debug/smoke_test_wan_mask.py \
    --model-path "$MODEL_PATH" \
    --height 480 \
    --width 736 \
    --frames 17

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Smoke test完成"
    echo "  查看上面的输出获取mask_config"
    echo "  下一步: 创建npo_masked_wan.yaml配置文件"
else
    echo "[FAILED] Smoke test失败 (exit code: $EXIT_CODE)"
    echo "  检查错误日志: slurm/logs/smoke_test_mask-$SLURM_JOB_ID.err"
fi
echo "Time: $(date)"
echo "========================================"

exit $EXIT_CODE
