#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-50
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -o slurm/logs/gen-safesora-%j.out
#SBATCH -e slurm/logs/gen-safesora-%j.err
# 用法: sbatch -J gen-ss-<method>-<shard> slurm/gen_safesora.sbatch <method> <shard> <num_shards>

set -euo pipefail
METHOD=$1; SHARD=$2; NSHARDS=$3

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u scripts/generate_safesora.py \
    --method "$METHOD" --shard "$SHARD" --num_shards "$NSHARDS"

# 脚本正常跑完（该分片所有视频已存在）才会到这里
touch "outputs/safesora_gen/DONE_${METHOD}_${SHARD}"
echo "DONE ${METHOD} shard ${SHARD}/${NSHARDS}"
