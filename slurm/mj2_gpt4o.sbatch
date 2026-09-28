#!/bin/bash
#SBATCH --job-name=mj2-gpt4o
#SBATCH --partition=berzelius
# 纯 API 调用不用 GPU, 但 berzelius partition 强制要求 --gres
# ⚠ 申请 GPU 却零利用率会在整 1h 被 admin 回收 (见 memory/experiments.md Exp010)
#   1025 条视频约需 25 分钟, 在 1h 墙以内; 若扩大规模务必分片
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=outputs/mj2_gpt4o_%j.log
#SBATCH --error=outputs/mj2_gpt4o_%j.err

# Exp 011: 五方法生成视频 · GPT-4o-mini · 统一 taxonomy
# 只换 --label_defs_path, 判别器脚本本身零改动
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
source ~/.bashrc
conda activate main
set -e

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

LABEL_DEFS="classify/configs/unified_label_definitions.json"
GEN_ROOT="outputs/safesora_gen"
OUT_ROOT="outputs/safesora_gen/multi_judge_v2"
METHODS=(base lora quant distill malicious)

for method in "${METHODS[@]}"; do
  VID="$GEN_ROOT/$method"
  if [ ! -d "$VID" ]; then echo "SKIP $VID (not found)"; continue; fi
  echo "=== [GPT-4o-mini · unified taxonomy] $method ==="
  python -u -m classify.evaluation.gpt4o_safety_judge \
    --video_dir "$VID" \
    --output_json "$OUT_ROOT/$method/gpt4o.json" \
    --num_frames 4 \
    --model gpt-4o-mini \
    --label_defs_path "$LABEL_DEFS" \
    --tag "mj2_gpt4om_4f_unified_${method}"
done

echo "EXIT_CODE=$?"
