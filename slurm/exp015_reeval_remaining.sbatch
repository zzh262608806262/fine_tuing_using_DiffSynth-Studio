#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 01:00:00
#SBATCH -o logs/exp015_reeval_remaining_%j.out
#SBATCH -e logs/exp015_reeval_remaining_%j.err

cd /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio

echo "=========================================="
echo "Exp015 NudeNet 重新评估 (剩余臂)"
echo "=========================================="
echo ""

# 只评估未完成的臂
arms=("npo_erased" "grad_ascent_base" "grad_ascent_erased" "anchor_distill_base" "anchor_distill_erased")

for arm_name in "${arms[@]}"; do
    video_dir="outputs/exp015/${arm_name}"
    output_file="outputs/exp015_evaluation/${arm_name}_evaluation.json"

    echo "=== 评估 $arm_name ==="
    echo "视频目录: $video_dir"

    python scripts/exp018_evaluate.py \
        --method "$arm_name" \
        --video_dir "$video_dir" \
        --output_file "$output_file" \
        --nudenet_threshold 0.6

    if [ $? -eq 0 ]; then
        echo "✓ $arm_name 评估完成"
        jq '.nudenet.summary' "$output_file"
    else
        echo "✗ $arm_name 评估失败"
    fi
    echo ""
done

echo "=========================================="
echo "所有评估完成，完整汇总:"
echo "=========================================="
for method in esd npo grad_ascent anchor_distill; do
    for arm in base erased; do
        arm_name="${method}_${arm}"
        output_file="outputs/exp015_evaluation/${arm_name}_evaluation.json"
        printf "%-25s " "$arm_name:"
        jq -r '"\(.nudenet.summary.violations)/\(.nudenet.summary.total_videos) (\(.nudenet.summary.violation_rate))"' "$output_file"
    done
done
