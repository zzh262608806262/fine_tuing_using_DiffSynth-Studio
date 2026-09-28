#!/usr/bin/env bash
# Exp012 快速验证版本提交脚本
# 目标：几小时内跑通整个流程，验证方法可行性
#
# 优化点：
# 1. 擦除训练：1200步 -> 200步
# 2. 微调：repeat=10 epochs=2 -> repeat=1 epochs=1 (20次曝光 -> 1次曝光)
# 3. 生成：只生成 fast 集 29条（跳过 benchmark 99条）
# 4. 预计总耗时：几小时（而不是3-5天）
#
# 用法: bash slurm/exp012_submit_quick.sh

set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
mkdir -p slurm/logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="slurm/logs/exp012_quick_pipeline_jobs.tsv"
[ -f "$LOG" ] && cp "$LOG" "$LOG.bak_$TS"
: > "$LOG"

echo "=== Exp012 快速验证版提交 $TS ==="
echo "优化：擦除200步 | 微调1次曝光 | 仅生成fast集29条"
echo ""

# J0: 下载模型（如果已存在会跳过）
J0=$(sbatch --parsable slurm/wan5b_download.sbatch)
echo -e "J0\tdownload\t$J0" >> "$LOG"; echo "J0 download=$J0"

# J1: 桥转换验证
J1=$(sbatch --parsable --dependency=afterok:$J0 slurm/wan5b_bridge_verify.sbatch)
echo -e "J1\tverify\t$J1" >> "$LOG"; echo "J1 verify=$J1"

# J1c: 基线视频重生成
J1C=$(sbatch --parsable --dependency=afterok:$J1 slurm/wan5b_baseline_regen.sbatch)
echo -e "J1c\tbaseline\t$J1C" >> "$LOG"; echo "J1c baseline=$J1C"

# J2: 擦除训练（快速版：200步）
J2=$(sbatch --parsable --export=ALL,RUN_PROBE=1,UNLEARN_STEPS=200,SAVE_EVERY=100 \
     --dependency=afterok:$J1C slurm/wan5b_unlearn.sbatch)
echo -e "J2\tunlearn-quick\t$J2" >> "$LOG"; echo "J2 unlearn-quick=$J2 (200步)"

# J3: 擦除 LoRA 合并
J3=$(sbatch --parsable --dependency=afterok:$J2 slurm/wan5b_merge_bridge.sbatch)
echo -e "J3\tmerge\t$J3" >> "$LOG"; echo "J3 merge=$J3"

# J4: 微调（快速版：repeat=1 epochs=1）
J4=$(sbatch --parsable --export=ALL,QUICK_MODE=1 \
     --dependency=afterok:$J3 slurm/wan5b_finetune_quick.sbatch)
echo -e "J4\tfinetune-quick\t$J4" >> "$LOG"; echo "J4 finetune-quick=$J4 (1次曝光)"

# GEN: 只生成 fast 集 29条（4个臂并行）
GEN=()
for arm in base erased base_ft erased_ft; do
  g=$(ARM="$arm" SET=fast BEGIN=0 END=29 \
      sbatch --parsable --dependency=afterok:$J4 slurm/wan5b_eval_gen.sbatch)
  GEN+=("$g")
  echo -e "J_\tgen:$arm\t$g" >> "$LOG"; echo "GEN $arm=$g (fast 29条)"
done

# J9: 判别
DEP="afterok:${GEN[0]}"
for g in "${GEN[@]:1}"; do DEP="$DEP:$g"; done
J9=$(JUDGE_SETS=fast sbatch --parsable --dependency=$DEP slurm/wan5b_eval_judge.sbatch)
echo -e "J9\tjudge\t$J9" >> "$LOG"; echo "J9 judge=$J9"

echo ""
echo "=== 快速验证版提交完成: $LOG ==="
echo "预计耗时：几小时（vs 完整版3-5天）"
echo "监控: squeue --me"
echo "      watch -n 60 'squeue --me'"
