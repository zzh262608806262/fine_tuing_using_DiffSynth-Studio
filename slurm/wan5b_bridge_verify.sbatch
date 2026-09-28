#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH -t 08:00:00
#SBATCH -J wan5b-bridge-verify
#SBATCH -o slurm/logs/wan5b_bridge_verify-%j.out
#SBATCH -e slurm/logs/wan5b_bridge_verify-%j.err
# Exp012 Task1 网关：diffusers 5B 基座 -> 桥转换 DiffSynth 原格式，并做「同 prompt+seed 双端生成等价」验证。
# 顺序：(1) CPU 桥转换 base  -> models/Wan-AI/Wan2.2-TI2V-5B/（含 T5/VAE 软链）
#       (2) DiffSynth 侧生成 2 条（diffsynth env）
#       (3) VU diffusers 侧生成同 2 条（VU .venv python 直调）
#       (4) 帧级相似度比对，cosine < 0.85 时退出码 1（下游 --dependency=afterok 自动停链）
# 提示词均为良性（data/wan5b/bridge_equiv_prompts.jsonl）。
set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

# 确保使用正确的 Python 环境和 C++ 库
export PATH="/home/x_jiage/.conda/envs/diffsynth/bin:$PATH"
export LD_LIBRARY_PATH="/home/x_jiage/.conda/envs/diffsynth/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "=== Python 环境验证 ==="
which python
python --version
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
python -c "import pandas; print('pandas:', pandas.__version__)" || { echo "ERROR: pandas 未找到！" >&2; exit 1; }

# 注意：不设置 PYTHONNOUSERSITE=1，否则会屏蔽 conda 环境的 site-packages
# export PYTHONNOUSERSITE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VU_REPO="${VU_REPO:-/home/x_jiage/jiage/video-unlearning}"
VU_PY="${VU_PY:-$VU_REPO/.venv/bin/python}"
PROMPTS="data/wan5b/bridge_equiv_prompts.jsonl"
BASE_OUT="models/Wan-AI/Wan2.2-TI2V-5B"
EQUIV_DIR="outputs/wan5b_bridge_equiv"
SHARED="models/DiffSynth-Studio/Wan-Series-Converted-Safetensors"
# 原生 DiffSynth 5B DiT（DiffSynth-Studio checkout，只读；用于桥产物逐键比对）
NATIVE_5B_DIT="/home/x_jiage/jiage/DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B"

# 定位 diffusers 5B transformer（J0 已重新下载到 FT 项目内，本节点应为单候选）
LOCAL_DIFFUSERS="$PWD/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
BASE_TRANSFORMER=""
for c in \
  "${WAN5B_BASE_TRANSFORMER:-}" \
  "$LOCAL_DIFFUSERS/transformer" \
  ; do
  if [ -n "$c" ] && [ -d "$c" ]; then BASE_TRANSFORMER="$c"; break; fi
done
[ -n "$BASE_TRANSFORMER" ] || { echo "[error] 无法定位 5B diffusers transformer（J0 下载目标 $LOCAL_DIFFUSERS/transformer 缺失）" >&2; exit 1; }
[ -f "$LOCAL_DIFFUSERS/model_index.json" ] || { echo "[error] J0 下载不完整: 缺 $LOCAL_DIFFUSERS/model_index.json" >&2; exit 1; }
echo "=== host: $(hostname) === base transformer: $BASE_TRANSFORMER"

echo "=== [1/4] CPU 桥转换 base -> $BASE_OUT ==="
mkdir -p "$BASE_OUT"
python3 scripts/convert_wan_5b_bridge.py \
    --input-dir "$BASE_TRANSFORMER" \
    --output-dir "$BASE_OUT" \
    --dtype auto --strict --check-hash

echo "=== [1.5/4] 原生 DiT 逐键比对闸门（桥产物 vs 原生 DiffSynth 5B DiT，相对容差门控）==="
python3 scripts/verify_bridge_native.py \
    --converted-dir "$BASE_OUT" \
    --native-dir "$NATIVE_5B_DIT" \
    --rel-tol 1e-2 \
    --outjson "$EQUIV_DIR/native_compare.json"

echo "=== [2/4] 软链 T5/VAE（共享 safetensors，绝对目标避免相对链接断裂）==="
for f in models_t5_umt5-xxl-enc-bf16.safetensors Wan2.2_VAE.safetensors; do
  if [ ! -e "$BASE_OUT/$f" ]; then ln -sf "$PWD/$SHARED/$f" "$BASE_OUT/$f"; fi
  ls -l "$BASE_OUT/$f" | awk '{print "  linked:", $NF}'
done

echo "=== [3/4] 双端同 seed 生成 ==="
mkdir -p "$EQUIV_DIR"
python -u scripts/verify_bridge_equiv.py gen-diffsynth "$PROMPTS" "$EQUIV_DIR/diffsynth"
[ -x "$VU_PY" ] || { echo "[error] VU venv python 不存在: $VU_PY" >&2; exit 1; }
"$VU_PY" -u scripts/verify_bridge_equiv.py gen-vu "$PROMPTS" "$EQUIV_DIR/vu" \
    --vu-root "$VU_REPO" --model-path "$LOCAL_DIFFUSERS"

echo "=== [4/4] 帧级相似度比对（门控: cosine < 0.80 -> 退出码 1）==="
"$VU_PY" -u scripts/verify_bridge_equiv.py compare \
    "$EQUIV_DIR/diffsynth" "$EQUIV_DIR/vu" "$PROMPTS" \
    --threshold 0.80 --outjson "$EQUIV_DIR/compare.json"

echo "DONE wan5b bridge verify (base converted + equivalence PASS)"