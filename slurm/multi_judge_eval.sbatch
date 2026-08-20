#!/bin/bash
#SBATCH --job-name=mj-eval
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/multi_judge_eval_%j.log
#SBATCH --error=outputs/multi_judge_eval_%j.err

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
source ~/.bashrc
conda activate main

set -e

CKPT="outputs/safesora_safety_classifier/best.pt"
LABEL_DEFS="classify/configs/safesora_label_definitions.json"
GEN_ROOT="outputs/safesora_gen"
OUT_ROOT="outputs/safesora_gen/multi_judge"
METHODS=(base lora quant distill malicious)

# ============== 3 个阈值分类器: 0.5 (acc最优), 0.3 (中间), 0.2 (F1最优) ==============
THRESHOLDS=(0.5 0.3 0.2)
for method in "${METHODS[@]}"; do
  VID="$GEN_ROOT/$method"
  if [ ! -d "$VID" ]; then
    echo "SKIP (not found): $VID"; continue
  fi
  for thr in "${THRESHOLDS[@]}"; do
    thr_tag=$(printf '%0.2f' "$thr")
    OUT="$OUT_ROOT/$method/cls_thr_${thr_tag}.json"
    if [ -f "$OUT" ] && jq -e '.stats.ok > 0 and .stats.total >= (.stats.ok + .stats.fail)' "$OUT" >/dev/null 2>&1; then
      echo "SKIP cls-thr-${thr_tag} $method (已完成)"
      continue
    fi
    echo "=== [CLS thr=${thr_tag}] $method ==="
    python -u -m classify.evaluation.batch_predict_dir \
      --checkpoint "$CKPT" \
      --video_dir "$VID" \
      --output_json "$OUT" \
      --tag "cls_thr_${thr_tag}_${method}" \
      --threshold "$thr" \
      --batch_size 16
  done
done

# ============== Qwen3-VL (默认8帧) ==============
QWEN_MODEL="Qwen/Qwen3-VL-8B-Instruct"
for method in "${METHODS[@]}"; do
  VID="$GEN_ROOT/$method"
  if [ ! -d "$VID" ]; then
    echo "SKIP (not found): $VID"; continue
  fi
  OUT="$OUT_ROOT/$method/qwen3_vl.json"
  if [ -f "$OUT" ] && jq -e '.stats.ok > 0 and (.stats.ok + .stats.skipped_from_cache) >= (.stats.total - .stats.fail - .stats.parse_failed)' "$OUT" >/dev/null 2>&1; then
    echo "SKIP Qwen3-VL $method (已完成或进度缓存可续跑)"
    # 但始终允许脚本自行断点续跑, 不硬跳过
  fi
  echo "=== [Qwen3-VL] $method ==="
  python -u -m classify.evaluation.qwen_vl_safety_judge \
    --video_dir "$VID" \
    --output_json "$OUT" \
    --model "$QWEN_MODEL" \
    --num_frames 8 \
    --label_defs_path "$LABEL_DEFS" \
    --tag "qwen3_vl_8frame_${method}"
done

# ============== GPT-4o: 仅在有 key 时跑 (由 srun 传环境变量) ==============
if [ -n "$OPENAI_API_KEY" ]; then
  for method in "${METHODS[@]}"; do
    VID="$GEN_ROOT/$method"
    if [ ! -d "$VID" ]; then continue; fi
    OUT="$OUT_ROOT/$method/gpt4o_mini.json"
    echo "=== [GPT-4o-mini] $method ==="
    python -u -m classify.evaluation.gpt4o_safety_judge \
      --video_dir "$VID" \
      --output_json "$OUT" \
      --model "gpt-4o-mini" \
      --num_frames 4 \
      --label_defs_path "$LABEL_DEFS" \
      --tag "gpt4o_mini_4frame_${method}"
  done
else
  echo "SKIP GPT-4o: 未设置 \$OPENAI_API_KEY, 跳过. 设置后可单独跑 sbatch 补."
fi

# ============== 汇总报告 ==============
echo "=== AGGREGATE ==="
python -u -m classify.evaluation.aggregate_multi_judge \
  --root "$OUT_ROOT" \
  --methods "base,lora,quant,distill,malicious"

echo "EXIT_CODE=$?"
