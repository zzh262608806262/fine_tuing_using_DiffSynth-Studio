#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH -t 04:00:00
#SBATCH -J wan5b-finetune-quick
#SBATCH -o slurm/logs/wan5b-finetune-quick-%j.out
#SBATCH -e slurm/logs/wan5b-finetune-quick-%j.err
# Wan2.2-TI2V-5B 良性微调 - 快速验证版
# 优化：repeat=1 epochs=1 (1次曝光，而非完整版的20次)
# 用于快速验证流程，不追求最佳微调效果

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 使用 VU 的 uv venv（与 J0-J3 一致，避免 transformers/torch 版本冲突）
VU_REPO="${VU_REPO:-/home/x_jiage/jiage/video-unlearning}"
VU_VENV_ACTIVATE="${VU_VENV_ACTIVATE:-$VU_REPO/.venv/bin/activate}"

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
if [ -f "$VU_VENV_ACTIVATE" ]; then
  echo "[env] 使用 VU uv venv: $VU_VENV_ACTIVATE"
  source "$VU_VENV_ACTIVATE"
else
  echo "[env] 未找到 VU venv($VU_VENV_ACTIVATE), 兜底 conda activate diffsynth"
  conda activate diffsynth
fi

export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0

# 添加 DiffSynth-Studio 到 PYTHONPATH，让 train.py 能 import diffsynth
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# 快速模式参数
REPEAT=1
EPOCHS=1

echo "=== 快速验证模式：repeat=$REPEAT epochs=$EPOCHS (总曝光=${REPEAT}×${EPOCHS}=1次) ==="

# 第一臂：擦除后基座 + FT
echo "=== 臂1/2: erased+FT ==="
python -u scripts/finetune_wan5b.py --arm erased \
    --dataset_repeat $REPEAT --num_epochs $EPOCHS
echo "DONE arm=erased"
ls -lh models/train/Wan2.2-TI2V-5B_erased_lora10e2/

# 第二臂：未擦除基座 + FT
echo "=== 臂2/2: base+FT ==="
python -u scripts/finetune_wan5b.py --arm base \
    --dataset_repeat $REPEAT --num_epochs $EPOCHS
echo "DONE arm=base"
ls -lh models/train/Wan2.2-TI2V-5B_base_lora10e2/

echo "ALL DONE wan5b quick fine-tune (1次曝光验证版)"
