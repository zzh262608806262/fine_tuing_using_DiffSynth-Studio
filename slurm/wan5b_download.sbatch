#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --partition=berzelius-cpu
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH -J wan5b-download
#SBATCH -o slurm/logs/wan5b_download-%j.out
#SBATCH -e slurm/logs/wan5b_download-%j.err
# Exp012 J0：重新下载 Wan-AI/Wan2.2-TI2V-5B-Diffusers（diffusers 布局，34.2GB，HF 官方受限仓库）。
# 背景：VU 配置钉定的 /scratch/s6398820/hf_cache/... 已随集群 /scratch 清空消失。
# 目标路径（FT 项目内）：models/Wan-AI/Wan2.2-TI2V-5B-Diffusers/
#   其中 transformer/ 供 VU 训练与 Task1/3 桥转换；text_encoder|tokenizer|vae|scheduler|model_index 供 VU cache/generate。
# 注意：Wan 模型库为受限仓库，须带 HF_TOKEN（外传自登录环境，勿硬编码）。
set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
VU_PY="${VU_PY:-/home/x_jiage/jiage/video-unlearning/.venv/bin/python}"
if conda activate diffsynth 2>/dev/null && python3 -c "import huggingface_hub" 2>/dev/null; then
  echo "[env] conda diffsynth + huggingface_hub OK"
  PY="python3"
else
  echo "[env] diffsynth env 不可用，兜底 VU venv python ($VU_PY)"
  PY="$VU_PY"
  "$PY" -c "import huggingface_hub, sys; print('hf hub', huggingface_hub.__version__)"
fi

export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
[ -n "${HF_TOKEN:-}" ] || { echo "[error] HF_TOKEN 未设置（受限仓库需鉴权）" >&2; exit 1; }

DEST="models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
export DEST_ABS="$PWD/$DEST"
mkdir -p "$(dirname "$DEST")"

echo "=== [1/2] snapshot_download Wan-AI/Wan2.2-TI2V-5B-Diffusers -> $DEST ==="
"$PY" -u - <<'PY'
import os
from huggingface_hub import snapshot_download

repo = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
dest = os.environ["DEST_ABS"]
print(f"[download] repo={repo} dest={dest}", flush=True)
path = snapshot_download(
    repo_id=repo,
    local_dir=dest,
    local_dir_use_symlinks=False,
    max_workers=8,
)
print(f"[download] done -> {path}", flush=True)
PY

echo "=== [2/2] 完整性核对 ==="
# 注：该仓库为扁平 diffusers 布局（model_index.json + transformer|text_encoder|vae|tokenizer|scheduler），
# 此前核对误把 _class_name/_diffusers_version（字符串）当组件目录而断言失败，现已改为按真实组件核查。
"$PY" - <<'PY'
import json, os
from pathlib import Path
dest = Path(os.environ["DEST_ABS"])
mi = dest / "model_index.json"
assert mi.is_file(), f"缺 model_index.json: {mi}"
comps = json.loads(mi.read_text())
assert comps.get("_class_name") == "WanPipeline", f"model_index 非 WanPipeline: {comps.get('_class_name')}"
expect = {"transformer", "text_encoder", "vae", "tokenizer", "scheduler"}
got = {p.name for p in dest.iterdir() if p.is_dir()}
missing = expect - got
assert not missing, f"缺组件目录: {sorted(missing)}"
print("components:", sorted(expect))
tr = sorted(p.name for p in (dest / "transformer").glob("diffusion_pytorch_model*.safetensors"))
assert len(tr) == 5, f"transformer 分片 != 5: {tr}"
te = sorted(p.name for p in (dest / "text_encoder").glob("model*.safetensors"))
assert len(te) == 3, f"text_encoder 分片 != 3: {te}"
vae = sorted(p.name for p in (dest / "vae").glob("*.safetensors"))
assert len(vae) == 1, f"vae 文件 != 1: {vae}"
assert (dest / "tokenizer" / "spiece.model").is_file(), "缺 tokenizer/spiece.model"
assert (dest / "scheduler" / "scheduler_config.json").is_file(), "缺 scheduler/scheduler_config.json"
print("OK: transformer shards =", tr)
print("OK: text_encoder shards =", te)
print("OK: vae =", vae)
print("OK: tokenizer + scheduler 配置齐全")
PY

echo "DONE wan5b download"