#!/bin/bash
#SBATCH --job-name=mj-gpt4o
#SBATCH --partition=berzelius
# GPT-4o 是 API 调用 + 视频抽帧, 不需要 GPU; berzelius partition 强制 GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/multi_judge_gpt4o_%j.log
#SBATCH --error=outputs/multi_judge_gpt4o_%j.err

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
source ~/.bashrc
conda activate main

set -e

echo "=== API smoke test ==="
python - <<'PY'
import os, sys
try:
    from openai import OpenAI
    c = OpenAI()
    r = c.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':'reply with just: OK'}], max_tokens=10, temperature=0, timeout=30)
    print(f"[smoke] API OK: {r.choices[0].message.content}")
except Exception as e:
    print(f"[smoke] API FAIL: {type(e).__name__}: {e}")
    sys.exit(1)
PY

LABEL_DEFS="classify/configs/safesora_label_definitions.json"
GEN_ROOT="outputs/safesora_gen"
OUT_ROOT="outputs/safesora_gen/multi_judge"
GPT_MODEL="gpt-4o-mini"
METHODS=(base lora quant distill malicious)

for method in "${METHODS[@]}"; do
  VID="$GEN_ROOT/$method"
  if [ ! -d "$VID" ]; then echo "SKIP $VID (not found)"; continue; fi
  OUT="$OUT_ROOT/$method/gpt4o.json"
  echo "=== [GPT-4o] $method ==="
  python -u -m classify.evaluation.gpt4o_safety_judge \
    --video_dir "$VID" \
    --output_json "$OUT" \
    --num_frames 4 \
    --model "$GPT_MODEL" \
    --label_defs_path "$LABEL_DEFS" \
    --tag "gpt4o_mini_4frame_${method}"
done

echo "=== AGGREGATE (更新) ==="
python -u -m classify.evaluation.aggregate_multi_judge \
  --root "$OUT_ROOT" \
  --methods "base,lora,quant,distill,malicious"

echo "EXIT_CODE=$?"
