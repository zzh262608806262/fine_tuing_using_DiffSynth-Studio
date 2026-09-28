#!/bin/bash
# Exp012 实验监控脚本
# 功能：实时监控作业状态、检测异常、记录日志、生成报告

set -euo pipefail

# ============ 配置 ============
BASE_DIR="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
LOG_DIR="$BASE_DIR/slurm/logs"
MONITOR_LOG="$BASE_DIR/outputs/exp012_monitor.log"
STATUS_FILE="$BASE_DIR/outputs/exp012_status.json"
REPORT_FILE="$BASE_DIR/outputs/exp012_report.md"
ALERT_FILE="$BASE_DIR/outputs/exp012_alerts.log"

# 从最新的 pipeline_jobs.tsv 读取作业 ID
JOBS_TSV="$BASE_DIR/slurm/logs/exp012_pipeline_jobs.tsv"

# 监控间隔（秒）
INTERVAL=${MONITOR_INTERVAL:-60}

# ============ 辅助函数 ============
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"
}

alert() {
    echo "[ALERT $(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$ALERT_FILE"
    log "⚠️  ALERT: $*"
}

# 读取作业 ID
read_job_ids() {
    if [ ! -f "$JOBS_TSV" ]; then
        log "错误：找不到 $JOBS_TSV"
        return 1
    fi
    
    # 跳过表头，读取最后一次提交的作业
    tail -n 11 "$JOBS_TSV" | while IFS=$'\t' read -r stage jid; do
        echo "$stage:$jid"
    done
}

# 获取作业状态
get_job_status() {
    local jid=$1
    sacct -j "$jid" --format=JobID,State,ExitCode,Elapsed,NodeList -P 2>/dev/null | grep "^$jid|" || echo "$jid|UNKNOWN|?|?|?"
}

# 检查作业健康状态
check_job_health() {
    local stage=$1
    local jid=$2
    local state=$3
    local exit_code=$4
    
    case "$state" in
        RUNNING)
            # 检查是否超时（可选）
            return 0
            ;;
        COMPLETED)
            if [ "$exit_code" = "0:0" ]; then
                log "✅ $stage ($jid) 成功完成"
                return 0
            else
                alert "$stage ($jid) 完成但退出码非零: $exit_code"
                return 1
            fi
            ;;
        FAILED)
            alert "$stage ($jid) 失败！退出码: $exit_code"
            # 尝试提取错误信息
            local err_log="$LOG_DIR/wan5b_*-$jid.err"
            if ls $err_log 2>/dev/null; then
                log "错误日志摘要:"
                tail -20 $err_log | head -10 >> "$MONITOR_LOG"
            fi
            return 1
            ;;
        CANCELLED|TIMEOUT)
            alert "$stage ($jid) 被取消或超时: $state"
            return 1
            ;;
        PENDING)
            # 检查依赖是否失败
            return 0
            ;;
        *)
            log "⚠️  $stage ($jid) 未知状态: $state"
            return 0
            ;;
    esac
}

# 生成状态 JSON
generate_status_json() {
    local timestamp=$(date -Iseconds)
    cat > "$STATUS_FILE" <<EOF
{
  "experiment": "Exp012",
  "timestamp": "$timestamp",
  "jobs": [
EOF
    
    local first=true
    read_job_ids | while IFS=: read -r stage jid; do
        local info=$(get_job_status "$jid")
        IFS='|' read -r _ state exit_code elapsed node <<< "$info"
        
        [ "$first" = true ] && first=false || echo "," >> "$STATUS_FILE"
        
        cat >> "$STATUS_FILE" <<EOF
    {
      "stage": "$stage",
      "job_id": "$jid",
      "state": "$state",
      "exit_code": "$exit_code",
      "elapsed": "$elapsed",
      "node": "$node"
    }
EOF
    done
    
    cat >> "$STATUS_FILE" <<EOF

  ]
}
EOF
}

# 生成报告
generate_report() {
    cat > "$REPORT_FILE" <<EOF
# Exp012 实验监控报告

**生成时间**: $(date '+%Y-%m-%d %H:%M:%S')

## 作业状态总览

| 阶段 | Job ID | 状态 | 退出码 | 耗时 | 节点 |
|------|--------|------|--------|------|------|
EOF
    
    read_job_ids | while IFS=: read -r stage jid; do
        local info=$(get_job_status "$jid")
        IFS='|' read -r _ state exit_code elapsed node <<< "$info"
        
        local status_icon
        case "$state" in
            COMPLETED) status_icon="✅" ;;
            RUNNING) status_icon="🔄" ;;
            PENDING) status_icon="⏳" ;;
            FAILED) status_icon="❌" ;;
            *) status_icon="⚠️" ;;
        esac
        
        echo "| $stage | $jid | $status_icon $state | $exit_code | $elapsed | $node |" >> "$REPORT_FILE"
    done
    
    cat >> "$REPORT_FILE" <<EOF

## 最近告警

EOF
    
    if [ -f "$ALERT_FILE" ]; then
        tail -10 "$ALERT_FILE" >> "$REPORT_FILE" || echo "无告警" >> "$REPORT_FILE"
    else
        echo "无告警" >> "$REPORT_FILE"
    fi
    
    log "报告已生成: $REPORT_FILE"
}

# 主监控循环
monitor_loop() {
    log "========================================="
    log "开始监控 Exp012 实验"
    log "监控间隔: ${INTERVAL}秒"
    log "========================================="
    
    local consecutive_failures=0
    local all_completed=false
    
    while [ "$all_completed" = false ]; do
        local running_count=0
        local pending_count=0
        local completed_count=0
        local failed_count=0
        
        log "--- 检查点 $(date '+%H:%M:%S') ---"
        
        read_job_ids | while IFS=: read -r stage jid; do
            local info=$(get_job_status "$jid")
            IFS='|' read -r _ state exit_code elapsed node <<< "$info"
            
            case "$state" in
                RUNNING) ((running_count++)) ;;
                PENDING) ((pending_count++)) ;;
                COMPLETED) ((completed_count++)) ;;
                FAILED) ((failed_count++)) ;;
            esac
            
            check_job_health "$stage" "$jid" "$state" "$exit_code" || ((consecutive_failures++))
        done
        
        log "运行中: $running_count | 等待: $pending_count | 完成: $completed_count | 失败: $failed_count"
        
        # 生成状态文件
        generate_status_json
        
        # 每小时生成一次报告
        local minute=$(date '+%M')
        if [ "$minute" = "00" ] || [ $failed_count -gt 0 ]; then
            generate_report
        fi
        
        # 检查是否全部完成
        if [ $running_count -eq 0 ] && [ $pending_count -eq 0 ]; then
            all_completed=true
            log "========================================="
            log "所有作业已完成！"
            log "完成: $completed_count | 失败: $failed_count"
            log "========================================="
            generate_report
            break
        fi
        
        # 检查连续失败
        if [ $consecutive_failures -ge 3 ]; then
            alert "检测到连续失败，请人工介入！"
            generate_report
            # 可选：发送通知、自动重试等
        fi
        
        sleep "$INTERVAL"
    done
}

# 一次性状态检查（非循环模式）
check_once() {
    log "执行一次性状态检查"
    
    read_job_ids | while IFS=: read -r stage jid; do
        local info=$(get_job_status "$jid")
        IFS='|' read -r _ state exit_code elapsed node <<< "$info"
        
        printf "%-20s %10s  %-12s  %8s  %10s  %s\n" \
            "$stage" "$jid" "$state" "$exit_code" "$elapsed" "$node"
        
        check_job_health "$stage" "$jid" "$state" "$exit_code" || true
    done
    
    generate_status_json
    generate_report
}

# ============ 主入口 ============
case "${1:-monitor}" in
    monitor)
        monitor_loop
        ;;
    check)
        check_once
        ;;
    report)
        generate_report
        cat "$REPORT_FILE"
        ;;
    status)
        generate_status_json
        cat "$STATUS_FILE"
        ;;
    *)
        echo "用法: $0 {monitor|check|report|status}"
        echo "  monitor - 持续监控（默认）"
        echo "  check   - 一次性检查"
        echo "  report  - 生成并显示报告"
        echo "  status  - 生成并显示状态 JSON"
        exit 1
        ;;
esac
