#!/bin/bash
#SBATCH --job-name=st-gpt4o
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/safesora_test_gpt4o_%j.log
#SBATCH --error=outputs/safesora_test_gpt4o_%j.err

# Exp 010: GPT-4o-mini on SafeSora test 集 5745 条 (4 帧, 与 Exp009 一致)
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
source ~/.bashrc
conda activate main

set -e

# Smoke test: 计算节点能否访问 OpenAI API
echo "=== API smoke test ==="
python - <<'PY'
import sys
try:
    from openai import OpenAI
    c = OpenAI()
    r = c.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':'reply with just: OK'}], max_tokens=10, temperature=0, timeout=30)
    print(f"[smoke] API OK: {r.choices[0].message.content}")
except Exception as e:
    print(f"[smoke] API FAIL: {type(e).__name__}: {e}")
    sys.exit(1)
PY

VIDEO_DIR="outputs/safesora_test_videos"
OUT_ROOT="outputs/safesora_test_multi_judge/test"

echo "=== [gpt4o] 4 frames ==="
python -u -m classify.evaluation.gpt4o_safety_judge \
  --video_dir "$VIDEO_DIR" \
  --output_json "$OUT_ROOT/gpt4o.json" \
  --suffix "*.mp4" \
  --num_frames 4 \
  --model gpt-4o-mini \
  --tag "safesora_test_gpt4om_4f"

echo "EXIT_CODE=$?"