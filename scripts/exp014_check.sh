#!/usr/bin/env bash
# Exp014 快速修复脚本集合

set -e

cd "$(dirname "$(dirname "$(readlink -f "$0")")")" # Auto: project root

JOB_ID="17460858"
LOG_FILE="slurm/logs/exp014_finetune-${JOB_ID}.out"
ERR_FILE="slurm/logs/exp014_finetune-${JOB_ID}.err"

# ============================================================================
# 函数：检查作业状态
# ============================================================================
check_job_status() {
    echo "📊 作业状态检查"
    echo "-----------------------------------"
    squeue -j $JOB_ID || echo "作业不在队列中"
    echo ""
}

# ============================================================================
# 函数：查看最新日志
# ============================================================================
view_logs() {
    echo "📄 最新日志（输出）"
    echo "-----------------------------------"
    if [ -f "$LOG_FILE" ]; then
        tail -50 "$LOG_FILE"
    else
        echo "日志文件还未生成"
    fi
    echo ""

    echo "📄 最新日志（错误）"
    echo "-----------------------------------"
    if [ -f "$ERR_FILE" ]; then
        tail -50 "$ERR_FILE"
    else
        echo "错误日志为空"
    fi
    echo ""
}

# ============================================================================
# 函数：检查常见错误
# ============================================================================
check_common_errors() {
    echo "🔍 常见错误检查"
    echo "-----------------------------------"

    if [ ! -f "$LOG_FILE" ]; then
        echo "⏳ 日志文件还未生成，等待作业启动..."
        return
    fi

    # 检查 CUDA OOM
    if grep -q "CUDA out of memory" "$LOG_FILE" "$ERR_FILE" 2>/dev/null; then
        echo "❌ 发现 CUDA OOM 错误"
        echo "修复建议："
        echo "  1. 降低 batch size"
        echo "  2. 减少 gradient_accumulation_steps"
        echo "  3. 使用更小的模型或更少的 frames"
        return
    fi

    # 检查文件不存在
    if grep -q "FileNotFoundError\|No such file" "$LOG_FILE" "$ERR_FILE" 2>/dev/null; then
        echo "❌ 发现文件路径错误"
        grep -A 2 "FileNotFoundError\|No such file" "$LOG_FILE" "$ERR_FILE" 2>/dev/null | head -10
        return
    fi

    # 检查权限错误
    if grep -q "Permission denied" "$LOG_FILE" "$ERR_FILE" 2>/dev/null; then
        echo "❌ 发现权限错误"
        grep -A 2 "Permission denied" "$LOG_FILE" "$ERR_FILE" 2>/dev/null | head -10
        return
    fi

    # 检查 Python 错误
    if grep -q "Traceback\|Exception\|Error:" "$LOG_FILE" "$ERR_FILE" 2>/dev/null; then
        echo "❌ 发现 Python 错误"
        grep -B 2 -A 5 "Traceback\|Exception\|Error:" "$LOG_FILE" "$ERR_FILE" 2>/dev/null | head -20
        return
    fi

    echo "✅ 未发现常见错误"
}

# ============================================================================
# 函数：检查产物
# ============================================================================
check_outputs() {
    echo "📦 产物检查"
    echo "-----------------------------------"

    for arm in erased base; do
        output_dir="models/finetune/exp014_${arm}_ft"
        echo ""
        echo "臂: $arm"

        if [ -d "$output_dir" ]; then
            checkpoint_count=$(ls -1 "$output_dir"/epoch-*.safetensors 2>/dev/null | wc -l)
            echo "  路径: $output_dir"
            echo "  Checkpoint 数量: $checkpoint_count / 20"

            if [ "$checkpoint_count" -gt 0 ]; then
                echo "  最新的 checkpoint:"
                ls -lt "$output_dir"/epoch-*.safetensors 2>/dev/null | head -3
            fi
        else
            echo "  ⏳ 输出目录还未创建"
        fi
    done
    echo ""
}

# ============================================================================
# 主菜单
# ============================================================================
case "${1:-status}" in
    status)
        check_job_status
        ;;
    logs)
        view_logs
        ;;
    errors)
        check_common_errors
        ;;
    outputs)
        check_outputs
        ;;
    full)
        check_job_status
        check_common_errors
        check_outputs
        view_logs
        ;;
    *)
        echo "用法: $0 {status|logs|errors|outputs|full}"
        echo ""
        echo "  status  - 查看作业状态"
        echo "  logs    - 查看最新日志"
        echo "  errors  - 检查常见错误"
        echo "  outputs - 检查产物"
        echo "  full    - 完整检查（默认）"
        ;;
esac
