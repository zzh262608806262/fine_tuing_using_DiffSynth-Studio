#!/bin/bash
# 登录节点 watchdog: 保证训练一直有"载体"在跑, 直到 DONE/FAILED.
#   1. 若 DONE (训练完成) 或 FAILED (连续崩溃 5 次, 需人工排查) -> 退出
#   2. 若初始 srun 载体进程 (run_pid) 还活着 -> 等待
#   3. 若队列里已有 safesora-cls 作业 (RUNNING 或 PENDING) -> 等待
#   4. 否则 sbatch 提交新 GPU 作业续跑 (train.sbatch 内部从 last.pt 自动 resume)
# 用法: nohup bash watchdog.sh > /dev/null 2>&1 &   (在登录节点运行)
REPO=/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
OUT=$REPO/outputs/safesora_safety_classifier
mkdir -p "$OUT"
LOG=$OUT/watchdog.log

log() { echo "[watchdog] $(date '+%F %T') $*" >> "$LOG"; }

log "started on $(hostname), pid $$"
while true; do
    if [ -f "$OUT/DONE" ]; then
        log "DONE marker found, training complete. exiting."
        exit 0
    fi
    if [ -f "$OUT/FAILED" ]; then
        log "FAILED marker found (5 consecutive crashes). NOT resubmitting; needs manual inspection."
        exit 1
    fi
    # 初始载体: 从登录节点 srun --overlap 挂进交互 job 的进程
    if [ -f "$OUT/run_pid" ] && kill -0 "$(cat "$OUT/run_pid")" 2>/dev/null; then
        sleep 120
        continue
    fi
    # 已有续跑作业在跑或在排队
    if squeue --me -h -n safesora-cls 2>/dev/null | grep -q .; then
        sleep 120
        continue
    fi
    log "no alive training carrier found -> submitting new job"
    sbatch "$REPO/classify/scripts/train.sbatch" >> "$LOG" 2>&1
    sleep 180
done
