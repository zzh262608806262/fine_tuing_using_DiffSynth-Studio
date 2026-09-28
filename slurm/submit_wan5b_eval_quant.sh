#!/bin/bash
# 批量提交量化模型评测视频生成任务

cd /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio

SET=${1:-benchmark}  # 默认 benchmark 集（99条）

echo "批量提交量化评测生成任务"
echo "  SET: $SET"
echo ""

# 确定数量
if [ "$SET" = "fast" ]; then
    TOTAL=29
elif [ "$SET" = "benchmark" ]; then
    TOTAL=99
else
    echo "错误: 未知的 SET=$SET (可选: fast, benchmark)"
    exit 1
fi

echo "核心对比组（研究重点）:"
echo "------------------------"

# 1. base_quant (基线)
echo "提交 base_quant..."
JOB_ID=$(sbatch --export=ARM=base_quant,SET=$SET,BEGIN=0,END=$TOTAL slurm/wan5b_eval_quant_gen.sbatch | awk '{print $4}')
echo "  Job ID: $JOB_ID"

# 2. erased_quant (研究目标)
echo "提交 erased_quant..."
JOB_ID=$(sbatch --export=ARM=erased_quant,SET=$SET,BEGIN=0,END=$TOTAL slurm/wan5b_eval_quant_gen.sbatch | awk '{print $4}')
echo "  Job ID: $JOB_ID"

echo ""
echo "可选对比组（已量化但非研究重点）:"
echo "-----------------------------------"
read -p "是否提交 base_ft_quant 和 erased_ft_quant? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "提交 base_ft_quant..."
    JOB_ID=$(sbatch --export=ARM=base_ft_quant,SET=$SET,BEGIN=0,END=$TOTAL slurm/wan5b_eval_quant_gen.sbatch | awk '{print $4}')
    echo "  Job ID: $JOB_ID"

    echo "提交 erased_ft_quant..."
    JOB_ID=$(sbatch --export=ARM=erased_ft_quant,SET=$SET,BEGIN=0,END=$TOTAL slurm/wan5b_eval_quant_gen.sbatch | awk '{print $4}')
    echo "  Job ID: $JOB_ID"
fi

echo ""
echo "所有任务已提交，查看状态："
squeue -u x_jiage -o "%.18i %.9P %.30j %.8u %.8T %.10M %.6D %R" | grep wan5b_eval_quant
