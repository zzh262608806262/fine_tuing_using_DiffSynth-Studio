#!/bin/bash
# 批量提交所有4个arm的量化任务

cd /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio

MODE=${1:-infer}
METHOD=${2:-bitsandbytes_nf4}

echo "批量提交量化任务"
echo "  MODE: $MODE"
echo "  METHOD: $METHOD"
echo ""

for ARM in base erased base_ft erased_ft; do
    echo "提交 $ARM..."
    JOB_ID=$(sbatch --export=ARM=$ARM,MODE=$MODE,METHOD=$METHOD slurm/quantize_wan5b.sbatch | awk '{print $4}')
    echo "  Job ID: $JOB_ID"
done

echo ""
echo "所有任务已提交，查看状态："
squeue -u x_jiage -o "%.18i %.9P %.30j %.8u %.8T %.10M %.6D %R" | grep quantize
