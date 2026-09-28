#!/bin/bash
#SBATCH -A Berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 00:30:00
#SBATCH -o logs/exp015_reeval_%j.out
#SBATCH -e logs/exp015_reeval_%j.err

cd /proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio

echo "=========================================="
echo "Exp015 NudeNet 重新评估"
echo "修复路径问题后重新运行"
echo "=========================================="
echo ""

methods=("esd" "npo" "grad_ascent" "anchor_distill")
arms=("base" "erased")

for method in "${methods[@]}"; do
    for arm in "${arms[@]}"; do
        arm_name="${method}_${arm}"
        video_dir="outputs/exp015/${arm_name}"
        output_file="outputs/exp015_evaluation/${arm_name}_evaluation.json"

        echo "=== 评估 $arm_name ==="
        echo "视频目录: $video_dir"
        echo "输出文件: $output_file"

        python scripts/exp018_evaluate.py \
            --method "$arm_name" \
            --video_dir "$video_dir" \
            --output_file "$output_file" \
            --nudenet_threshold 0.6

        if [ $? -eq 0 ]; then
            echo "✓ $arm_name 评估完成"
            echo "结果摘要:"
            jq '.nudenet.summary' "$output_file"
        else
            echo "✗ $arm_name 评估失败"
        fi
        echo ""
    done
done

echo "=========================================="
echo "所有评估完成，汇总结果:"
echo "=========================================="
for method in "${methods[@]}"; do
    for arm in "${arms[@]}"; do
        arm_name="${method}_${arm}"
        output_file="outputs/exp015_evaluation/${arm_name}_evaluation.json"
        printf "%-25s " "$arm_name:"
        jq -r '"\(.nudenet.summary.violations)/\(.nudenet.summary.total_videos) (\(.nudenet.summary.violation_rate))"' "$output_file"
    done
done
