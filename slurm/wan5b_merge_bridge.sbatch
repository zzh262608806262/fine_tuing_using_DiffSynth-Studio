#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH -t 12:00:00
#SBATCH -J wan5b-merge-bridge
#SBATCH -o slurm/logs/wan5b_merge_bridge-%j.out
#SBATCH -e slurm/logs/wan5b_merge_bridge-%j.err

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface

# Task 2 擦除 LoRA（VU GradAscent，video-transformer-lora-v1 格式；可换成 adapter_step<N>.pt）
ADAPTER="models/unlearn/wan5b_nudity_grad_ascent/adapter_final.pt"
[ -f "$ADAPTER" ] || { echo "[error] 擦除 LoRA 缺失: $ADAPTER" >&2; exit 1; }
# 定位 diffusers 5B transformer（J0 重新下载到 FT 内，单候选）
LOCAL_DIFFUSERS="$PWD/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
BASE_TRANSFORMER=""
for c in \
  "${WAN5B_BASE_TRANSFORMER:-}" \
  "$LOCAL_DIFFUSERS/transformer" \
  ; do
  if [ -n "$c" ] && [ -d "$c" ]; then BASE_TRANSFORMER="$c"; break; fi
done
[ -n "$BASE_TRANSFORMER" ] || { echo "[error] 无法定位 5B diffusers transformer（J0 下载目标 $LOCAL_DIFFUSERS/transformer 缺失）" >&2; exit 1; }
echo "=== base transformer: $BASE_TRANSFORMER ==="
# 合并后 diffusers transformer 目录（"擦除后全量权重"）
MERGED_DIR="models/unlearn/wan5b_nudity_grad_ascent/merged"
# 桥转换后 DiffSynth 擦除后基座（T5/VAE 复用 models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/，由生成侧负责）
ERASED_DIR="models/Wan-AI/Wan2.2-TI2V-5B-erased"

echo "=== [1/x] 合并擦除 LoRA → ${MERGED_DIR}，并经 Task1 桥 → ${ERASED_DIR} ==="
python3 scripts/merge_unlearn_lora_wan5b.py \
    --adapter "$ADAPTER" \
    --base-dir "$BASE_TRANSFORMER" \
    --output-dir "$MERGED_DIR" \
    --bridge-output-dir "$ERASED_DIR" \
    --bridge-dtype bf16 \
    --max-shard-size-gb 4.0 \
    --strict --check-hash \
    --force

# 擦除后基座同样补齐 T5/VAE 软链（三件套齐备，供微调/生成使用）
SHARED="models/DiffSynth-Studio/Wan-Series-Converted-Safetensors"
mkdir -p "$ERASED_DIR"
for f in models_t5_umt5-xxl-enc-bf16.safetensors Wan2.2_VAE.safetensors; do
  if [ ! -e "$ERASED_DIR/$f" ]; then ln -sf "$PWD/$SHARED/$f" "$ERASED_DIR/$f"; fi
done

echo "=== [x/x] 产物核对 ==="
echo "--- merged diffusers transformer ---"
ls -la "$MERGED_DIR"
echo "--- DiffSynth 擦除后基座 ---"
ls -la "$ERASED_DIR"
echo "--- merge_report.json（摘要） ---"
python3 - <<'PY'
import json
with open("models/unlearn/wan5b_nudity_grad_ascent/merged/merge_report.json") as f:
    r = json.load(f)
print("summary:", json.dumps(r["summary"], ensure_ascii=False, indent=2))
print("scaling:", r["scaling"])
print("shard_rewrites:", [s["file"] for s in r["shard_rewrites"]])
if r["unmatched"]:
    print("unmatched:", r["unmatched"][:10])
PY

echo "DONE wan5b merge + bridge"