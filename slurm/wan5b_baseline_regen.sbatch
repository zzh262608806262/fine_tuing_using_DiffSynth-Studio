#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH -t 04:00:00
#SBATCH -J wan5b-baseline-regen
#SBATCH -o slurm/logs/wan5b_baseline_regen-%j.out
#SBATCH -e slurm/logs/wan5b_baseline_regen-%j.err
# Exp012 J1c：重生成 27 条 unlearn 基线视频（forget 18 + retain 9）并重写派生 manifest。
# 背景：VU compose_wan manifest 的 video 指向 /scratch/.../baseline（已随 /scratch 清空消失）；
#       这里用 Task1 桥产物（models/Wan-AI/Wan2.2-TI2V-5B，DiffSynth 原生基座）按 manifest
#       prompt+seed、统一 17 帧/480x720/50 步/cfg5/flow5 重生成，并经 VU NudeNet 闸门复核。
set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

# 确保使用正确的 C++ 库
export PATH="/home/x_jiage/.conda/envs/diffsynth/bin:$PATH"
export LD_LIBRARY_PATH="/home/x_jiage/.conda/envs/diffsynth/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VU_REPO="${VU_REPO:-/home/x_jiage/jiage/video-unlearning}"

echo "=== [1/2] 重生成基线视频 + NudeNet 闸门复核 + 派生 manifest ==="
python3 -u scripts/regen_unlearn_baseline_wan5b.py \
    --forget-manifest "$VU_REPO/data/splits/nudity_forget_composed_wan.jsonl" \
    --retain-manifest "$VU_REPO/data/splits/nudity_retain_composed_wan.jsonl" \
    --vu-root "$VU_REPO" \
    --frames 17 --height 480 --width 736 --steps 50 --cfg 5.0 --flow 5.0 --fps 16 \
    --out-dir data/wan5b/unlearn_baseline \
    --out-forget data/wan5b/unlearn_baseline_forget.jsonl \
    --out-retain data/wan5b/unlearn_baseline_retain.jsonl

echo "=== [2/2] 产物核对 ==="
echo "--- 视频 ---"
find data/wan5b/unlearn_baseline -name "*.mp4" | wc -l
echo "--- 派生 manifest 条数 ---"
wc -l data/wan5b/unlearn_baseline_forget.jsonl data/wan5b/unlearn_baseline_retain.jsonl
echo "--- 抽查 manifest video 字段 ---"
head -n 1 data/wan5b/unlearn_baseline_forget.jsonl | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['class_label'], r['cell'], r['video'], r.get('regen'))"

echo "DONE wan5b baseline regen"