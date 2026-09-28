#!/usr/bin/env bash
# Exp012 完整流水线一键提交（登录节点运行，无 GPU 需求）。
# 依赖链（slurm --dependency=afterok，任一步失败自动停链）:
#   J0  download        （重新下载 HF diffusers 5B 34.2GB → FT models/；原 /scratch 快照已清空）
#     -> J1 bridge-verify（基座桥转换 + 原生 DiT 逐键比对 + 同 seed 双端等价门控）
#     -> J1c baseline-regen（重生成 27 条 unlearn 基线视频 + NudeNet 闸门复核 + 派生 manifest）
#     -> J2 unlearn(cache + GradAscent 训练; RUN_PROBE=1 追加探针)
#     -> J3 merge-bridge(擦除 LoRA 合并 + 擦除后基座桥回 + T5/VAE 软链)
#     -> J4 finetune(两臂: erased+FT / base+FT)
#     -> J5-J8 eval-gen(四臂, 每臂单 job 全量 fast+benchmark)
#     -> J9 eval-judge(判别: 分类器 porn 多阈值 + NudeNet; abstain on missing)
# 用法: bash slurm/exp012_submit.sh   （会向 stdout 打印全部 job id，并落 slurm/logs/exp012_pipeline_jobs.tsv）
set -euo pipefail

cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
mkdir -p slurm/logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="slurm/logs/exp012_pipeline_jobs.tsv"
[ -f "$LOG" ] && cp "$LOG" "$LOG.bak_$TS"
: > "$LOG"

echo "=== Exp012 流水线提交 $TS ==="

J0=$(sbatch --parsable slurm/wan5b_download.sbatch)
echo -e "J0\tdownload\t$J0" >> "$LOG"; echo "J0 download=$J0"
J1=$(sbatch --parsable --dependency=afterok:$J0 slurm/wan5b_bridge_verify.sbatch)
echo -e "J1\tverify\t$J1" >> "$LOG"; echo "J1 verify=$J1"
J1C=$(sbatch --parsable --dependency=afterok:$J1 slurm/wan5b_baseline_regen.sbatch)
echo -e "J1c\tbaseline\t$J1C" >> "$LOG"; echo "J1c baseline=$J1C"
J2=$(sbatch --parsable --export=ALL,RUN_PROBE=1 --dependency=afterok:$J1C slurm/wan5b_unlearn.sbatch)
echo -e "J2\tunlearn\t$J2" >> "$LOG"; echo "J2 unlearn=$J2"
J3=$(sbatch --parsable --dependency=afterok:$J2 slurm/wan5b_merge_bridge.sbatch)
echo -e "J3\tmerge\t$J3" >> "$LOG"; echo "J3 merge=$J3"
J4=$(sbatch --parsable --dependency=afterok:$J3 slurm/wan5b_finetune.sbatch)
echo -e "J4\tfinetune\t$J4" >> "$LOG"; echo "J4 finetune=$J4"

GEN=()
for arm in base erased base_ft erased_ft; do
  g=$(ARM="$arm" SET=fast,benchmark BEGIN=0 END=999 \
      sbatch --parsable --dependency=afterok:$J4 slurm/wan5b_eval_gen.sbatch)
  GEN+=("$g")
  echo -e "J_\tgen:$arm\t$g" >> "$LOG"; echo "GEN $arm=$g"
done

DEP="afterok:${GEN[0]}"
for g in "${GEN[@]:1}"; do DEP="$DEP:$g"; done
J5=$(sbatch --parsable --dependency=$DEP slurm/wan5b_eval_judge.sbatch)
echo -e "J9\tjudge\t$J5" >> "$LOG"; echo "J9 judge=$J5"

echo "=== 全部提交完成: $LOG ==="
echo "squeue -u \$USER | grep -E 'wan5b|exp012|$(hostname)' 查看进度"
echo "监视: watch -n 60 'squeue -u \$USER'"