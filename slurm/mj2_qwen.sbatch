#!/bin/bash
#SBATCH --job-name=mj2-qwen
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/mj2_qwen_%j.log
#SBATCH --error=outputs/mj2_qwen_%j.err

# Exp 011: 五方法生成视频 · Qwen3-VL · 统一 taxonomy
# 只换 --label_defs_path, 判别器脚本本身零改动 (其他项目仍用旧配置)
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
module load buildenv-gcccuda/12.4.1-gcc13.3.0
source ~/.bashrc
conda activate main
set -e

LABEL_DEFS="classify/configs/unified_label_definitions.json"
GEN_ROOT="outputs/safesora_gen"
OUT_ROOT="outputs/safesora_gen/multi_judge_v2"
QWEN_MODEL="Qwen/Qwen3-VL-8B-Instruct"
METHODS=(base lora quant distill malicious)

for method in "${METHODS[@]}"; do
  VID="$GEN_ROOT/$method"
  if [ ! -d "$VID" ]; then echo "SKIP $VID (not found)"; continue; fi
  echo "=== [Qwen3-VL · unified taxonomy] $method ==="
  python -u -m classify.evaluation.qwen_vl_safety_judge \
    --video_dir "$VID" \
    --output_json "$OUT_ROOT/$method/qwen3_vl.json" \
    --model "$QWEN_MODEL" \
    --num_frames 8 \
    --label_defs_path "$LABEL_DEFS" \
    --tag "mj2_qwen3vl_8f_unified_${method}"
done

echo "EXIT_CODE=$?"
