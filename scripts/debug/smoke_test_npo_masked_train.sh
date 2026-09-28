#!/bin/bash
# NPOMasked 完整训练smoke test - 在交互节点运行
# 只跑1个step验证整个训练循环

set -e

echo "=========================================="
echo "NPOMasked 完整训练 Smoke Test"
echo "=========================================="

cd "$(dirname "$(dirname "$(readlink -f "$0")")")" # Auto: project root

# 激活DiffSynth环境
source /home/x_jiage/miniforge3/bin/activate
conda activate diffsynth

echo "[1/2] 运行1步训练..."
python scripts/run_unlearn_wan5b.py \
  --method NPOMasked \
  --erase-concept "breasts" \
  --model-path models/Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --manifest-root /home/x_jiage/jiage/video-unlearning/data/splits \
  --output-dir models/unlearned/smoke_test_npo_masked \
  --steps 1 \
  --save-interval 999999 \
  2>&1 | tee /tmp/npo_masked_smoke_test.log

echo ""
echo "[2/2] 检查训练是否成功..."
if grep -q "Step 1/1" /tmp/npo_masked_smoke_test.log; then
    echo "✅ 训练步骤执行成功"
else
    echo "❌ 未找到训练步骤标记"
    exit 1
fi

if grep -qi "error\|traceback\|failed" /tmp/npo_masked_smoke_test.log; then
    echo "❌ 发现错误"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅✅✅ 完整训练流程验证通过！"
echo "=========================================="
echo ""
echo "下一步: 取消当前作业，重新提交完整训练"
