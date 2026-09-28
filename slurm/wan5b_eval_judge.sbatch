#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH -t 24:00:00
#SBATCH -J wan5b-eval-judge
#SBATCH -o slurm/logs/wan5b_eval_judge-%j.out
#SBATCH -e slurm/logs/wan5b_eval_judge-%j.err
# Exp012 四臂 porn 评测判别（生成完成后跑）：SafetyPredictor porn@0.2/0.3/0.5 +
# VU NudeNet（只读 import）+ 可选 qwen3vl（本地 .cache/huggingface，默认 off）。
# 分区头/环境抄 slurm/gen_safesora.sbatch + wan5b 系列 -C fat（conda diffsynth、DIFFSYNTH_SKIP_DOWNLOAD=true）。
# 用法:
#   sbatch slurm/wan5b_eval_judge.sbatch
#   JUDGE_ARMS=base,erased sbatch slurm/wan5b_eval_judge.sbatch
#   JUDGE_NUDENET_OFF=1 sbatch slurm/wan5b_eval_judge.sbatch       # NudeNet 权重无法在线下载时
#   JUDGE_MODE=qwen3vl JUDGE_ARMS=base,erased sbatch slurm/wan5b_eval_judge.sbatch
# 可选环境变量: JUDGE_ARMS(默认四臂) JUDGE_SETS(默认 fast,benchmark) JUDGE_MODE(默认 off) JUDGE_NUDENET_OFF(默认空)

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
# 判别用 main env（与既有 slurm/wan5b_porn_eval.sbatch 先例一致：SafetyPredictor/NudeNet/decord 依赖在 main）
conda activate main

export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
# qwen3vl judge 只允许用本地缓存（--judge qwen3vl 时生效），禁止任何在线请求
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

JUDGE_ARMS="${JUDGE_ARMS:-base,erased,base_ft,erased_ft}"
JUDGE_SETS="${JUDGE_SETS:-fast,benchmark}"
JUDGE_MODE="${JUDGE_MODE:-off}"
JUDGE_NUDENET_OFF="${JUDGE_NUDENET_OFF:-}"

EXTRA_ARGS=()
if [ "$JUDGE_NUDENET_OFF" = "1" ]; then
    EXTRA_ARGS+=(--nudenet-off)
fi
if [ "$JUDGE_MODE" != "off" ]; then
    EXTRA_ARGS+=(--judge "$JUDGE_MODE")
fi

echo "=== 判别: arms=$JUDGE_ARMS sets=$JUDGE_SETS mode=$JUDGE_MODE ==="
python -u scripts/eval_porn_wan5b.py \
    --arm "$JUDGE_ARMS" \
    --set "$JUDGE_SETS" \
    "${EXTRA_ARGS[@]}"

echo "DONE judge wan5b eval"