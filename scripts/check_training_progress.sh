#!/bin/bash
# 持续检查 exp014 训练进度

JOB_ID="17461003"
LOG_FILE="slurm/logs/exp014_finetune_v2-${JOB_ID}.out"

echo "监控 Exp014 v2 训练进度..."
echo "Job ID: $JOB_ID"
echo "日志: $LOG_FILE"
echo ""

# 循环检查直到看到训练开始
for i in {1..10}; do
    echo "=== 检查 $i ($(date +%H:%M:%S)) ==="

    # 检查作业状态
    STATUS=$(squeue -j $JOB_ID -h -o "%T %M" 2>/dev/null || echo "NOT_FOUND")
    echo "作业状态: $STATUS"

    # 检查日志行数
    if [ -f "$LOG_FILE" ]; then
        LINES=$(wc -l < "$LOG_FILE")
        echo "日志行数: $LINES"

        # 查找关键信息
        if grep -q "Loading.*model" "$LOG_FILE" 2>/dev/null; then
            echo "✓ 正在加载模型..."
        fi

        if grep -q "Epoch" "$LOG_FILE" 2>/dev/null; then
            echo "✓ 训练已开始！"
            echo ""
            echo "最新进度:"
            grep "Epoch\|loss\|step" "$LOG_FILE" | tail -5
            exit 0
        fi

        if grep -iE "(error|exception|failed)" "$LOG_FILE" 2>/dev/null | tail -1; then
            echo "❌ 发现错误"
            exit 1
        fi
    fi

    echo ""
    sleep 30
done

echo "初始化时间较长，请继续等待..."
echo "查看实时日志: tail -f $LOG_FILE"
