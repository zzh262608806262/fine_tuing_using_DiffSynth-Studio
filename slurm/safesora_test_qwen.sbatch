#!/bin/bash
#SBATCH --job-name=st-qwen
#SBATCH --partition=berzelius
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/safesora_test_qwen_%j.log
#SBATCH --error=outputs/safesora_test_qwen_%j.err

# Exp 010: Qwen3-VL-8B on SafeSora test 集 5745 条 (8 帧)
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
# NSC 限制裸 gcc 调用; 加载 buildenv-gcccuda 以便 Qwen3-VL jit-compile CUDA 扩展
module load buildenv-gcccuda/12.4.1-gcc13.3.0
source ~/.bashrc
conda activate main

set -e

VIDEO_DIR="outputs/safesora_test_videos"
OUT_ROOT="outputs/safesora_test_multi_judge/test"

echo "=== [qwen3_vl] 8 frames ==="
python -u -m classify.evaluation.qwen_vl_safety_judge \
  --video_dir "$VIDEO_DIR" \
  --output_json "$OUT_ROOT/qwen3_vl.json" \
  --suffix "*.mp4" \
  --num_frames 8 \
  --tag "safesora_test_qwen3vl_8f"

echo "EXIT_CODE=$?"