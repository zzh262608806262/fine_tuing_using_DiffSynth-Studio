#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
# 5B BF16 训练 + 探针 VAE 解码用 fat 节点更稳; 若排队久可删掉 -C fat 让 40GB A100 也参与 backfill
# (VU exp504 的 5B 峰值能进 40GB A100)。
#SBATCH -C fat
#SBATCH -t 08:00:00
#SBATCH -J wan5b-unlearn
#SBATCH -o slurm/logs/wan5b-unlearn-%j.out
#SBATCH -e slurm/logs/wan5b-unlearn-%j.err

# =============================================================================
# Wan2.2-TI2V-5B nudity 遗忘作业:  缓存 latent -> 训练 (多方法支持) -> (可选)探针
#
# 用法(在 FT 仓库根目录下):
#   sbatch slurm/wan5b_unlearn.sbatch        (模块加载/日志目录写法照抄 lora100_rebuild.sbatch)
#
# 可覆盖的环境变量(默认值见下):
#   VU_REPO           VU(video-unlearning)仓库根目录(只读), 默认 /home/x_jiage/jiage/video-unlearning
#   FT_ROOT           FT 仓库根目录(所有产物落这里)
#   VU_VENV_ACTIVATE  VU 的 uv venv activate 脚本路径(默认 $VU_REPO/.venv/bin/activate)
#   VU_CONDA_ENV      兜底 conda env 名(默认 diffsynth)
#   WAN_MODEL_PATH    显式指定 Wan 快照路径; 留空则由脚本读 VU yaml 里的集群路径
#   CACHE_H/CACHE_W   缓存分辨率(默认 480x736, 参照 VU memory Exp505 5B Wan 基线)
#   METHOD            遗忘方法(默认 GradAscent): GradAscent | AnchorDistill | ESD | NPO
#   ANCHOR            AnchorDistill 专用: 替代词 (如 "a person") 或空字符串 "" (空 prompt)
#   NEGATIVE_GUIDANCE ESD 专用: 负引导强度 η (默认 1.0)
#   UNLEARN_STEPS/SAVE_EVERY  训练步数/存点间隔(默认 1200/200; exp504 的 GradAscent 用 600 步/200 存点,
#                            本项目 grad_ascent_wan.yaml 是 lora_rank=8/batch=1 更轻, Wan5B 经验量级 1000-2000 步)
#   RUN_PROBE         置 1 时训练后追加 base vs erased 探针生成
#
# 环境说明:
#   训练侧依赖以 VU pyproject.toml/uv.lock 锁定版本为准(hydra/omegaconf/diffusers 0.38 等), 建议跑在
#   VU 的 uv venv 下(exp504 slurm 即 source $VU_REPO/.venv/bin/activate); 找不到时兜底 FT 的
#   diffsynth conda env(本作业激活方式: module load Miniforge3 + conda activate, 照抄 FT 既有 sbatch)。
#   FT 侧 conda env 与 VU 依赖版本可能冲突, 优先 VU venv。
# =============================================================================
set -euo pipefail

VU_REPO="${VU_REPO:-/home/x_jiage/jiage/video-unlearning}"
FT_ROOT="${FT_ROOT:-/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio}"
VU_VENV_ACTIVATE="${VU_VENV_ACTIVATE:-$VU_REPO/.venv/bin/activate}"
VU_CONDA_ENV="${VU_CONDA_ENV:-diffsynth}"
# diffusers 5B 基座: 默认 J0 重新下载到 FT 内的 models/Wan-AI/Wan2.2-TI2V-5B-Diffusers
# （原 /scratch/s6398820 快照已随集群清空消失，VU yaml 钉定路径不可再用）
WAN_MODEL_PATH="${WAN_MODEL_PATH:-$FT_ROOT/models/Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
# 缓存输入: J1c 重生成的派生 manifest（FT 内，video 指向本地重生成基线）
CACHE_FORGET="${CACHE_FORGET:-$FT_ROOT/data/wan5b/unlearn_baseline_forget.jsonl}"
CACHE_RETAIN="${CACHE_RETAIN:-$FT_ROOT/data/wan5b/unlearn_baseline_retain.jsonl}"
CACHE_H="${CACHE_H:-480}"
CACHE_W="${CACHE_W:-736}"
METHOD="${METHOD:-GradAscent}"
ANCHOR="${ANCHOR:-}"
NEGATIVE_GUIDANCE="${NEGATIVE_GUIDANCE:-}"
UNLEARN_STEPS="${UNLEARN_STEPS:-1200}"
SAVE_EVERY="${SAVE_EVERY:-200}"
RUN_PROBE="${RUN_PROBE:-0}"

[ -d "$VU_REPO/src" ] || { echo "VU repo 无效(缺 src 包): $VU_REPO" >&2; exit 1; }
[ -d "$FT_ROOT" ] || { echo "FT root 无效: $FT_ROOT" >&2; exit 1; }

cd "$FT_ROOT"

# ---- 激活环境: 照抄 FT 既有 sbatch 的 Miniforge3/conda 写法; 优先 VU 的 uv venv ----
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
if [ -f "$VU_VENV_ACTIVATE" ]; then
  echo "[env] 使用 VU uv venv: $VU_VENV_ACTIVATE"
  # shellcheck disable=SC1090
  source "$VU_VENV_ACTIVATE"
else
  echo "[env] 未找到 VU venv($VU_VENV_ACTIVATE), 兜底 conda activate $VU_CONDA_ENV"
  conda activate "$VU_CONDA_ENV"
fi
export PYTHONNOUSERSITE=1

export HF_HOME="${HF_HOME:-/proj/berzelius-aiics-real/users/x_jiage/huggingface_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# VU 只读: 禁止 import 时向 VU/src 写 __pycache__ 字节码
export PYTHONDONTWRITEBYTECODE=1

mkdir -p slurm/logs

echo "=== host: $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

EXTRA_ARGS=()
[ -n "$WAN_MODEL_PATH" ] && EXTRA_ARGS=(--model-path "$WAN_MODEL_PATH")

echo "=== [1/2] 缓存 latent (forget+retain, AutoencoderKLWan + latents_mean/std) ==="
echo "     forget manifest = $CACHE_FORGET"
echo "     retain manifest = $CACHE_RETAIN"
for m in "$CACHE_FORGET" "$CACHE_RETAIN"; do
  [ -f "$m" ] || { echo "[error] 派生 manifest 缺失: $m" >&2; exit 1; }
done
python3 -u scripts/run_unlearn_wan5b.py \
  --vu-root "$VU_REPO" \
  --cache \
  --forget-manifest "$CACHE_FORGET" --retain-manifest "$CACHE_RETAIN" \
  --cache-height "$CACHE_H" --cache-width "$CACHE_W" \
  "${EXTRA_ARGS[@]}"

echo "=== [2/2] 训练 ($METHOD, Wan5B transformer LoRA, 只消费 latent manifest) ==="
TRAIN_ARGS=(--method "$METHOD")
if [ "$METHOD" = "AnchorDistill" ]; then
  [ -n "$ANCHOR" ] || { echo "[error] AnchorDistill 需要 ANCHOR 参数" >&2; exit 1; }
  TRAIN_ARGS+=(--anchor "$ANCHOR")
  echo "     anchor = $ANCHOR"
elif [ "$METHOD" = "ESD" ]; then
  [ -n "$NEGATIVE_GUIDANCE" ] && TRAIN_ARGS+=(--negative-guidance "$NEGATIVE_GUIDANCE")
  echo "     negative_guidance = ${NEGATIVE_GUIDANCE:-1.0 (default)}"
fi
python3 -u scripts/run_unlearn_wan5b.py \
  --vu-root "$VU_REPO" \
  --train \
  "${TRAIN_ARGS[@]}" \
  --unlearn-steps "$UNLEARN_STEPS" --save-every "$SAVE_EVERY" \
  "${EXTRA_ARGS[@]}"

if [ "$RUN_PROBE" = "1" ]; then
  echo "=== [3] 探针生成 base vs erased (同一批 nudity prompt / 同 seed) ==="
  python3 -u scripts/run_unlearn_wan5b.py \
    --vu-root "$VU_REPO" \
    --probe \
    "${EXTRA_ARGS[@]}"
fi

echo "=== all stages done ==="