#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 00:30:00
#SBATCH -J exp020a_debug
#SBATCH -o slurm/logs/exp020a_debug-%j.out
#SBATCH -e slurm/logs/exp020a_debug-%j.err

set -e

echo "=========================================="
echo "NPOMasked Debug - 打印shape信息"
echo "=========================================="

cd /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio

# 激活VU环境
VU_ROOT="/home/x_jiage/jiage/video-unlearning"
source "$VU_ROOT/.venv/bin/activate"

# 临时修改attention_mask.py添加debug输出
cp "$VU_ROOT/src/unlearning/training/attention_mask.py" "$VU_ROOT/src/unlearning/training/attention_mask.py.bak"

# 在masked_mean的第192行后添加debug
cat > /tmp/patch_debug.py << 'PYEOF'
import sys
original_file = sys.argv[1]

with open(original_file, 'r') as f:
    lines = f.readlines()

# 在line 192后插入debug
new_lines = []
for i, line in enumerate(lines, 1):
    new_lines.append(line)
    if i == 192:
        new_lines.append('    print(f"[DEBUG] sq.shape={sq.shape}, mask.shape={mask.shape}, b={b}, fr={fr}")\n')
        new_lines.append('    print(f"[DEBUG] m.shape after unsqueeze={m.shape}, m.numel()={m.numel()}")\n')

with open(original_file, 'w') as f:
    f.writelines(new_lines)
PYEOF

python /tmp/patch_debug.py "$VU_ROOT/src/unlearning/training/attention_mask.py"

echo "已添加debug输出，运行训练..."

# 运行1步训练
python scripts/run_unlearn_wan5b.py \
  --train \
  --method NPOMasked \
  --erase-concept "breasts" \
  --model-path models/Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --forget-manifest data/splits/nudity_forget_composed_wan.jsonl \
  --train-dir models/unlearned/smoke_test_npo_masked \
  --unlearn-steps 1 \
  --save-every 999999

# 恢复原文件
mv "$VU_ROOT/src/unlearning/training/attention_mask.py.bak" "$VU_ROOT/src/unlearning/training/attention_mask.py"

echo ""
echo "完成时间: $(date)"
