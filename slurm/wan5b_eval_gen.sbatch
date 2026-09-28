#!/usr/bin/env bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -C fat
#SBATCH -t 3-00:00:00
#SBATCH -J wan5b-eval-gen
#SBATCH -o slurm/logs/wan5b_eval_gen-%j.out
#SBATCH -e slurm/logs/wan5b_eval_gen-%j.err
# Exp012 四臂 Wan2.2-TI2V-5B porn 评测视频生成（分区头/环境抄 slurm/gen_safesora.sbatch + wan5b 系列 -C fat）
# 单臂单 job，用 ARM/BEGIN/END 拆区间并行提交多份；断点续跑（已存在 mp4 自动跳过、先写 tmp 再原子改名）。
# 用法（fast=29 条 idx 0..28，benchmark=99 条 idx 0..98；BEGIN/END 为半开区间 [BEGIN, END)）:
#   ARM=base  BEGIN=0  END=29  sbatch slurm/wan5b_eval_gen.sbatch    # fast 全集
#   ARM=base  BEGIN=29 END=99  sbatch slurm/wan5b_eval_gen.sbatch    # benchmark 后 70 条
#   ARM=erased_ft SET=benchmark BEGIN=0 END=99 sbatch slurm/wan5b_eval_gen.sbatch
# 四臂分批提交示例:
#   for arm in base erased base_ft erased_ft; do
#     ARM=$arm BEGIN=0  END=29  sbatch slurm/wan5b_eval_gen.sbatch
#     ARM=$arm BEGIN=29 END=99  sbatch slurm/wan5b_eval_gen.sbatch
#   done
# 可选环境变量: ARM(必填) BEGIN(默认0) END(默认-1=全量) SET(默认 fast,benchmark)

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

export PYTHONNOUSERSITE=1
# 只读本地 models/ 权重，禁止在线下载（ModelConfig 本地路径 + 显式 origin_file_pattern）
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARM="${ARM:?需设置 ARM=base|erased|base_ft|erased_ft}"
BEGIN="${BEGIN:-0}"
END="${END:--1}"
SET="${SET:-fast,benchmark}"

# 若 arm 目录缺 T5/VAE，软链共享 safetensors（仅当目标不存在；DiT 桥产物必须真实存在）
case "$ARM" in
    base|base_ft)     ARM_DIR="models/Wan-AI/Wan2.2-TI2V-5B" ;;
    erased|erased_ft) ARM_DIR="models/Wan-AI/Wan2.2-TI2V-5B-erased" ;;
    *) echo "[error] 未知 arm: $ARM"; exit 2 ;;
esac
SHARED="models/DiffSynth-Studio/Wan-Series-Converted-Safetensors"
mkdir -p "$ARM_DIR"
for f in models_t5_umt5-xxl-enc-bf16.safetensors Wan2.2_VAE.safetensors; do
    if [ ! -e "$ARM_DIR/$f" ]; then
        ln -sf "$PWD/$SHARED/$f" "$ARM_DIR/$f"
        echo "[link] $ARM_DIR/$f -> $PWD/$SHARED/$f"
    fi
done

echo "=== 生成: arm=$ARM set=$SET idx=[$BEGIN,$END) ==="
python -u scripts/generate_wan5b_eval.py \
    --arm "$ARM" --set "$SET" --start "$BEGIN" --end "$END"

# 脚本正常跑完（该分片所有视频已存在或已生成）才会到这里
echo "DONE gen wan5b eval arm=$ARM idx=[$BEGIN,$END)"