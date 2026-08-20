#!/bin/bash
#SBATCH --job-name=mj-qwen
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/multi_judge_qwen_%j.log
#SBATCH --error=outputs/multi_judge_qwen_%j.err

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
# NSC 限制裸 gcc 调用; 加载 buildenv-gcccuda 以便 Qwen3-VL jit-compile CUDA 扩展
module load buildenv-gcccuda/12.4.1-gcc13.3.0
source ~/.bashrc
conda activate main

set -e

LABEL_DEFS="classify/configs/safesora_label_definitions.json"
GEN_ROOT="outputs/safesora_gen"
OUT_ROOT="outputs/safesora_gen/multi_judge"
QWEN_MODEL="Qwen/Qwen3-VL-8B-Instruct"
METHODS=(base lora quant distill malicious)

for method in "${METHODS[@]}"; do
  VID="$GEN_ROOT/$method"
  if [ ! -d "$VID" ]; then echo "SKIP $VID (not found)"; continue; fi
  OUT="$OUT_ROOT/$method/qwen3_vl.json"
  echo "=== [Qwen3-VL] $method ==="
  python -u -m classify.evaluation.qwen_vl_safety_judge \
    --video_dir "$VID" \
    --output_json "$OUT" \
    --model "$QWEN_MODEL" \
    --num_frames 8 \
    --label_defs_path "$LABEL_DEFS" \
    --tag "qwen3_vl_8frame_${method}"
done

# GPT-4o 跳过（OPENAI_API_KEY 未设置，需用户后续单独跑）

echo "=== AGGREGATE (更新) ==="
python -u -m classify.evaluation.aggregate_multi_judge \
  --root "$OUT_ROOT" \
  --methods "base,lora,quant,distill,malicious"

echo "EXIT_CODE=$?"
