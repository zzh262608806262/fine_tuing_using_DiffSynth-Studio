#!/bin/bash
# SafeSora 三方法生成 watchdog（登录节点运行）:
# 保证每个 (method, shard) 任务始终有作业在跑/排队，直到 DONE。
# 崩溃保护: 若某任务连续 5 次重提交且视频数量无增长 -> 写 FAILED 标记并停止该任务。
# 用法: nohup bash slurm/gen_watchdog.sh > /dev/null 2>&1 &
REPO=/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
OUT=$REPO/outputs/safesora_gen
mkdir -p "$OUT"
LOG=$OUT/watchdog.log

# method shard num_shards（抽样 200 条模式：每方法单作业即可）
TASKS=(
    "base 0 1"
    "distill 0 1"
    "lora 0 1"
    "quant 0 1"
)

log() { echo "[gen-watchdog] $(date '+%F %T') $*" >> "$LOG"; }
log "started on $(hostname), pid $$"

while true; do
    all_done=1
    for task in "${TASKS[@]}"; do
        read -r m s n <<< "$task"
        key="${m}_${s}"
        [ -f "$OUT/DONE_$key" ] && continue
        if [ -f "$OUT/FAILED_$key" ]; then
            continue  # 需人工排查，不算 all_done 完成，但也不再重提
        fi
        all_done=0
        # 已有同名作业在跑或排队
        if squeue --me -h -n "gen-ss-$m-$s" 2>/dev/null | grep -q .; then
            continue
        fi
        # 崩溃计数: 与上次提交时相比视频数无增长则 +1
        count=$(ls "$OUT/$m" 2>/dev/null | grep -c '\.mp4$')
        last=$(cat "$OUT/.last_count_$key" 2>/dev/null || echo -1)
        fails=$(cat "$OUT/.fails_$key" 2>/dev/null || echo 0)
        if [ "$count" -le "$last" ]; then
            fails=$((fails + 1))
        else
            fails=0
        fi
        if [ "$fails" -ge 5 ]; then
            log "$key: 5 consecutive submits with no progress (count=$count) -> FAILED"
            touch "$OUT/FAILED_$key"
            continue
        fi
        echo "$count" > "$OUT/.last_count_$key"
        echo "$fails" > "$OUT/.fails_$key"
        log "$key: submitting (videos=$count, fails=$fails)"
        (cd "$REPO" && sbatch -J "gen-ss-$m-$s" slurm/gen_safesora.sbatch "$m" "$s" "$n") >> "$LOG" 2>&1
    done
    if [ "$all_done" -eq 1 ]; then
        log "all tasks DONE (or FAILED-marked). exiting."
        exit 0
    fi
    sleep 300
done
