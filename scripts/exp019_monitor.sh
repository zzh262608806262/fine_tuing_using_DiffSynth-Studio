#!/bin/bash
# Exp019 进度监控和结果对比脚本

echo "=== Exp019 GradDiff 实验进度监控 ==="
echo "检查时间: $(date)"
echo ""

# 检查擦除训练
echo "【Stage 1】擦除训练"
if [ -d "models/unlearn/exp019_wan5b_nudity_graddiff" ]; then
    echo "  ✅ 输出目录存在"
    checkpoint_count=$(ls models/unlearn/exp019_wan5b_nudity_graddiff/step-* 2>/dev/null | wc -l)
    echo "  Checkpoints: ${checkpoint_count}/7"
    if [ ${checkpoint_count} -eq 7 ]; then
        echo "  ✅ 擦除训练完成"
    fi
else
    echo "  ⏳ 擦除训练进行中"
fi

# 检查合并模型
echo ""
echo "【Stage 2】LoRA 合并"
for model in graddiff_base graddiff_erased; do
    if [ -d "models/wan5b/exp019_${model}" ]; then
        size=$(du -sh models/wan5b/exp019_${model} 2>/dev/null | cut -f1)
        echo "  ✅ ${model}: ${size}"
    else
        echo "  ⏳ ${model}: 待合并"
    fi
done

# 检查视频生成
echo ""
echo "【Stage 3-4】视频生成"
for arm in graddiff_base graddiff_erased graddiff_ft_e19; do
    if [ -d "outputs/exp019/${arm}" ]; then
        video_count=$(ls outputs/exp019/${arm}/*.mp4 2>/dev/null | wc -l)
        echo "  ${arm}: ${video_count}/99 视频"
    else
        echo "  ${arm}: 待生成"
    fi
done

# 检查评估结果
echo ""
echo "【Stage 5】评估结果"
for arm in graddiff_base graddiff_erased graddiff_ft_e19; do
    eval_file="outputs/exp019_evaluation/${arm}_evaluation.json"
    if [ -f "${eval_file}" ]; then
        vr=$(python3 -c "import json; data=json.load(open('${eval_file}')); print(f\"{data['nudenet']['summary']['violation_rate']:.1%}\")")
        echo "  ✅ ${arm}: 违规率=${vr}"
    else
        echo "  ⏳ ${arm}: 待评估"
    fi
done

# 对比 GradAscent vs GradDiff
echo ""
echo "【对比分析】GradAscent (Exp015/018) vs GradDiff (Exp019)"
echo ""

if [ -f "outputs/exp019_evaluation/graddiff_erased_evaluation.json" ] && \
   [ -f "outputs/exp015_evaluation/grad_ascent_erased_evaluation.json" ]; then

    echo "擦除效果对比:"
    python3 << 'EOF'
import json
import sys

try:
    # GradAscent (Exp015)
    ga_base = json.load(open('outputs/exp015_evaluation/grad_ascent_base_evaluation.json'))
    ga_erased = json.load(open('outputs/exp015_evaluation/grad_ascent_erased_evaluation.json'))

    # GradDiff (Exp019)
    gd_base = json.load(open('outputs/exp019_evaluation/graddiff_base_evaluation.json'))
    gd_erased = json.load(open('outputs/exp019_evaluation/graddiff_erased_evaluation.json'))

    print("  GradAscent: {:.1%} → {:.1%} (Δ{:+.1%})".format(
        ga_base['nudenet']['summary']['violation_rate'],
        ga_erased['nudenet']['summary']['violation_rate'],
        ga_erased['nudenet']['summary']['violation_rate'] - ga_base['nudenet']['summary']['violation_rate']
    ))

    print("  GradDiff:   {:.1%} → {:.1%} (Δ{:+.1%})".format(
        gd_base['nudenet']['summary']['violation_rate'],
        gd_erased['nudenet']['summary']['violation_rate'],
        gd_erased['nudenet']['summary']['violation_rate'] - gd_base['nudenet']['summary']['violation_rate']
    ))
except Exception as e:
    print(f"  (数据不完整: {e})")
EOF
else
    echo "  ⏳ 等待基线评估完成"
fi

echo ""

if [ -f "outputs/exp019_evaluation/graddiff_ft_e19_evaluation.json" ] && \
   [ -f "outputs/exp018/evaluation/grad_ascent_ft_e19_results.json" ]; then

    echo "微调后回潮对比:"
    python3 << 'EOF'
import json

try:
    # GradAscent (Exp015 + Exp018)
    ga_erased = json.load(open('outputs/exp015_evaluation/grad_ascent_erased_evaluation.json'))
    ga_ft = json.load(open('outputs/exp018/evaluation/grad_ascent_ft_e19_results.json'))

    # GradDiff (Exp019)
    gd_erased = json.load(open('outputs/exp019_evaluation/graddiff_erased_evaluation.json'))
    gd_ft = json.load(open('outputs/exp019_evaluation/graddiff_ft_e19_evaluation.json'))

    ga_rebound = ga_ft['nudenet']['summary']['violation_rate'] - ga_erased['nudenet']['summary']['violation_rate']
    gd_rebound = gd_ft['nudenet']['summary']['violation_rate'] - gd_erased['nudenet']['summary']['violation_rate']

    print("  GradAscent 回潮: {:+.1%}".format(ga_rebound))
    print("  GradDiff 回潮:   {:+.1%}".format(gd_rebound))
    print("")

    if gd_rebound < ga_rebound:
        print("  ✅ 结论: GradDiff 更鲁棒 (retain 约束有效)")
    else:
        print("  ❌ 结论: GradDiff 不如 GradAscent 鲁棒 (retain 约束无效)")
except Exception as e:
    print(f"  (数据不完整: {e})")
EOF
else
    echo "  ⏳ 等待微调评估完成"
fi

echo ""
echo "=== 监控结束 ==="
