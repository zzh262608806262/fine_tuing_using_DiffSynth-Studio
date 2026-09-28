#!/bin/bash
# 批量保存所有4个arm的量化checkpoint

cd /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio

METHOD=${1:-bitsandbytes_nf4}

echo "批量保存量化checkpoint"
echo "  METHOD: $METHOD"
echo ""

for ARM in base erased base_ft erased_ft; do
    echo "提交 $ARM (save模式)..."
    JOB_ID=$(sbatch --export=ARM=$ARM,MODE=save,METHOD=$METHOD slurm/quantize_wan5b.sbatch | awk '{print $4}')
    echo "  Job ID: $JOB_ID"
done

echo ""
echo "所有保存任务已提交，查看状态："
squeue -u x_jiage -o "%.18i %.9P %.30j %.8u %.8T %.10M %.6D %R" | grep quantize
