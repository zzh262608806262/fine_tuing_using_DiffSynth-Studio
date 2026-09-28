#!/bin/bash
# Exp 015 多方法对比实验 - 全流程提交脚本
#
# 流程设计:
#   Stage 1: 擦除训练（4个方法并行）
#   Stage 2: LoRA 合并（4个方法并行，依赖 Stage 1）
#   Stage 3: 微调（4个方法并行，依赖 Stage 2）
#   Stage 4: 生成评测视频（8个臂并行，依赖 Stage 3）
#   Stage 5: 评估（统一评估，依赖 Stage 4）
#
# 复用 Exp 012 资源: base/base_ft 的生成和评估结果

set -euo pipefail

FT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
cd "$FT_ROOT"

echo "========================================"
echo "Exp 015 多方法对比实验 - 提交作业"
echo "Time: $(date)"
echo "========================================"

# 检查前置条件
echo "[CHECK] 检查前置条件..."

# 1. 检查 Exp 012 的 base 桥产物
BASE_MODEL="models/Wan-AI/Wan2.2-TI2V-5B"
if [ ! -d "$BASE_MODEL" ]; then
    echo "  [ERROR] 缺少 base 桥产物: $BASE_MODEL"
    echo "  请先运行 Exp 012 或执行桥转换"
    exit 1
fi
echo "  ✓ base 桥产物存在"

# 2. 检查 latent 缓存
LATENT_CACHE="data/wan5b/unlearn_latents"
if [ ! -d "$LATENT_CACHE" ]; then
    echo "  [ERROR] 缺少 latent 缓存: $LATENT_CACHE"
    echo "  请先运行缓存阶段"
    exit 1
fi
LATENT_COUNT=$(find "$LATENT_CACHE" -name "*.pt" | wc -l)
echo "  ✓ Latent 缓存存在 ($LATENT_COUNT 个文件)"

# 3. 检查 sbatch 脚本
METHODS=("grad_diff" "esd" "npo" "anchor_distill")
for method in "${METHODS[@]}"; do
    if [ ! -f "slurm/exp015_unlearn_${method}.sbatch" ]; then
        echo "  [ERROR] 缺少 sbatch 脚本: slurm/exp015_unlearn_${method}.sbatch"
        exit 1
    fi
done
echo "  ✓ 所有 sbatch 脚本存在"

# 创建日志目录和作业记录文件
mkdir -p slurm/logs
JOB_LOG="slurm/logs/exp015_pipeline_jobs.tsv"
echo -e "stage\tmethod\tjob_id" > "$JOB_LOG"

echo ""
echo "========================================"
echo "Stage 1: 擦除训练（4个方法并行）"
echo "========================================"

declare -A UNLEARN_JOBS

for method in "${METHODS[@]}"; do
    echo "[SUBMIT] $method 擦除训练..."
    JID=$(sbatch --parsable "slurm/exp015_unlearn_${method}.sbatch")
    UNLEARN_JOBS[$method]=$JID
    echo "  Job ID: $JID"
    echo -e "unlearn\t$method\t$JID" >> "$JOB_LOG"
done

echo ""
echo "已提交 ${#UNLEARN_JOBS[@]} 个擦除训练作业"
echo "作业记录已保存到: $JOB_LOG"

echo ""
echo "========================================"
echo "Stage 2-5: 待实现"
echo "========================================"
echo ""
echo "当前仅提交了 Stage 1（擦除训练）"
echo "后续阶段需要等待 Stage 1 完成后手动提交，或使用依赖链自动提交"
echo ""
echo "监控作业状态:"
echo "  squeue -u \$USER"
echo "  sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList"
echo ""
echo "查看作业日志:"
for method in "${METHODS[@]}"; do
    jid=${UNLEARN_JOBS[$method]}
    echo "  tail -f slurm/logs/exp015_unlearn_${method}-${jid}.out"
done

echo ""
echo "========================================"
echo "提交完成"
echo "========================================"

cat "$JOB_LOG"
