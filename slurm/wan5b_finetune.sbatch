#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH -t 3-00:00:00
#SBATCH -J wan5b-finetune
#SBATCH -o slurm/logs/wan5b-finetune-%j.out
#SBATCH -e slurm/logs/wan5b-finetune-%j.err
# Wan2.2-TI2V-5B 良性微调（两臂对照，文本驱动）
#   erased : 基座 models/Wan-AI/Wan2.2-TI2V-5B-erased/（Task3 产物，擦除已合进基座）
#   base   : 基座 models/Wan-AI/Wan2.2-TI2V-5B/       （Task1 产物）
# 两臂共用同一份 metadata 与冻结超参（见 scripts/finetune_wan5b.py 默认值：
#   repeat=10 epochs=2 lr=1e-4 rank=32 480x832 num_frames=25，曝光=10*2=20 次/样本），仅基座不同。
# 本文件在一个 job 内串行跑两臂；如需拆开分别提交，保留其中一行 --arm 即可。

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

export PYTHONNOUSERSITE=1
# 只读本地 models/ 下权重，禁止在线下载（5B 定位 = 本地路径 + 显式 origin_file_pattern）
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 两臂共用同一数据顺序（train.py 未暴露 --seed，用固定 hash seed 减少随机差异）
export PYTHONHASHSEED=0

# 第一臂：擦除后基座 + FT
python -u scripts/finetune_wan5b.py --arm erased
echo "DONE arm=erased"
ls -la models/train/Wan2.2-TI2V-5B_erased_lora10e2/

# 第二臂：未擦除基座 + FT（对照）
python -u scripts/finetune_wan5b.py --arm base
echo "DONE arm=base"
ls -la models/train/Wan2.2-TI2V-5B_base_lora10e2/

echo "ALL DONE wan5b fine-tune (erased + base)"