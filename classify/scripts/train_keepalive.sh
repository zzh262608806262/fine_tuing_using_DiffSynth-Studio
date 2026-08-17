#!/bin/bash
# 内层守护: 训练进程异常退出时自动重启 (从 last.pt 续跑), 最多重试 5 次.
# 正常完成 -> 写 DONE 标记; 连续失败放弃 -> 写 FAILED 标记 (watchdog 据此停止重新提交).
# 结点被回收时本脚本随 job 一起结束, 由登录节点的 watchdog.sh 负责重新申请结点.
REPO=/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
OUT=$REPO/outputs/safesora_safety_classifier
mkdir -p "$OUT"

MAX_RETRY=5
n=0
while true; do
    echo "[keepalive] $(date '+%F %T') host=$(hostname) starting attempt $((n+1))" >> "$OUT/keepalive.log"
    bash "$REPO/classify/scripts/run_train.sh" >> "$OUT/train_stdout.log" 2>&1
    code=$?
    if [ $code -eq 0 ]; then
        echo "[keepalive] $(date '+%F %T') training finished normally" >> "$OUT/keepalive.log"
        touch "$OUT/DONE"
        break
    fi
    n=$((n+1))
    echo "[keepalive] $(date '+%F %T') training exited with code $code (attempt $n/$MAX_RETRY)" >> "$OUT/keepalive.log"
    if [ $n -ge $MAX_RETRY ]; then
        echo "[keepalive] giving up after $MAX_RETRY retries" >> "$OUT/keepalive.log"
        touch "$OUT/FAILED"
        break
    fi
    sleep 30
done
