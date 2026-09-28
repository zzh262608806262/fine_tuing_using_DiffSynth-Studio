#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --partition=berzelius
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH -t 00:30:00
#SBATCH -J wan5b-vu-smoke
#SBATCH -o slurm/logs/wan5b_vu_smoke-%j.out
#SBATCH -e slurm/logs/wan5b_vu_smoke-%j.err
# Exp012 诊断：VU diffusers 管线 + 新下载 0.35.0.dev0 仓库（z_dim=48 3D VAE，spatial scale 16）
# 验证 latent 偶数尺寸(480x736→46)能完整走通 prepare_latent→predict_noise→scheduler.step，
# 对照 480x720→45(奇数，预期被 transformer patch 截断为 44)。不改 VU 仓库。
set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
VU_PY="${VU_PY:-/home/x_jiage/jiage/video-unlearning/.venv/bin/python}"
export PYTHONNOUSERSITE=1

DIFFUSERS="models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
VU_ROOT="/home/x_jiage/jiage/video-unlearning"

"$VU_PY" -u - <<PY
import sys
sys.path.insert(0, "$VU_ROOT")
import torch
from omegaconf import OmegaConf
from src.models.wan import WanModel

device = "cuda"
model = WanModel(
    model_path="$DIFFUSERS", device=device, dtype=torch.bfloat16,
    load_text_encoder=True, load_transformer=True, load_vae=True,
    load_scheduler=True, gradient_checkpointing=False,
)
print("VAE z_dim:", model.vae.config.z_dim, "scale_factor_spatial:", model.vae.config.scale_factor_spatial)
print("transformer in_channels:", model.denoiser.config.in_channels, "patch:", model.denoiser.config.patch_size)

emb = model.encode_text("a red car parked on a quiet street in the evening, cinematic")

def probe(tag, h, w, nf=17):
    latent = model.prepare_latent(nf, h, w)
    print(f"[{tag}] latent shape = {tuple(latent.shape)}")
    model.scheduler.set_timesteps(50, device=device)
    t = model.scheduler.timesteps[0]
    noise = model.predict_noise(latent, t, emb)
    print(f"[{tag}] model_output shape = {tuple(noise.shape)}", flush=True)
    if noise.shape == latent.shape:
        out = model.scheduler.step(noise, t, latent).prev_sample
        print(f"[{tag}] scheduler.step OK -> {tuple(out.shape)}")
        return True
    print(f"[{tag}] SHAPE MISMATCH (latent {tuple(latent.shape)} vs output {tuple(noise.shape)}) -> 预期失败路径")
    return False

ok720 = probe("480x720", 480, 720)
ok736 = probe("480x736", 480, 736)
print("RESULT 480x720:", "PASS(不改也行)" if ok720 else "FAIL(奇数 latent 破坏)", flush=True)
print("RESULT 480x736:", "PASS" if ok736 else "FAIL", flush=True)
sys.exit(0 if ok736 else 2)
PY

echo "DONE wan5b vu smoke"