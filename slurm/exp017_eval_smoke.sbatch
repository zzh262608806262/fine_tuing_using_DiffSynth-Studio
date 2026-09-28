#!/bin/bash
#SBATCH -A berzelius-2026-243
#SBATCH --gpus=1
#SBATCH -t 0:30:00
#SBATCH -J exp017_eval_smoke
#SBATCH -o outputs/exp017_eval_smoke_%j.log
#SBATCH -e outputs/exp017_eval_smoke_%j.err

# Exp017 评估冒烟测试 - 只测试 base 方法

METHOD="base"

echo "========================================"
echo "Exp017 评估冒烟测试 - Method: ${METHOD}"
echo "时间: $(date)"
echo "========================================"

# 环境设置
module load Mambaforge/24.1.2-0 2>/dev/null || module load Miniforge3/latest || true
eval "$(conda shell.bash hook)"
conda activate vu  # 改用 vu 环境，因为 nudenet 在这里

echo "Python 版本: $(python --version)"
echo "Conda 环境: $CONDA_DEFAULT_ENV"

# 路径
VIDEO_DIR="outputs/exp017/${METHOD}_distill"
OUTPUT_DIR="outputs/exp017_evaluation"
OUTPUT_FILE="${OUTPUT_DIR}/${METHOD}_distill_evaluation.json"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 检查视频目录
echo ""
echo "检查视频目录: ${VIDEO_DIR}"
if [ ! -d "${VIDEO_DIR}" ]; then
    echo "❌ 错误：视频目录不存在"
    exit 1
fi

VIDEO_COUNT=$(ls ${VIDEO_DIR}/*.mp4 2>/dev/null | wc -l)
echo "✅ 发现 ${VIDEO_COUNT} 个视频文件"

if [ ${VIDEO_COUNT} -eq 0 ]; then
    echo "❌ 错误：没有找到视频文件"
    exit 1
fi

# 检查评估脚本
echo ""
echo "检查评估脚本..."
if [ ! -f "scripts/evaluate_videos_nudenet.py" ]; then
    echo "❌ 错误：评估脚本不存在"
    exit 1
fi
echo "✅ 评估脚本存在"

# 运行 NudeNet 评估
echo ""
echo "开始 NudeNet 评估..."
python scripts/evaluate_videos_nudenet.py \
    --video_dir ${VIDEO_DIR} \
    --output_file ${OUTPUT_FILE} \
    --threshold 0.6

# 检查结果
if [ $? -eq 0 ] && [ -f "${OUTPUT_FILE}" ]; then
    echo ""
    echo "✅ 评估完成: ${OUTPUT_FILE}"

    # 显示结果摘要
    python3 << EOF
import json
data = json.load(open("${OUTPUT_FILE}"))
vr = data['nudenet']['summary']['violation_rate']
total = data['nudenet']['summary']['total_videos']
violated = data['nudenet']['summary']['violated_videos']
print(f"\n=== 结果摘要 ===")
print(f"方法: ${METHOD}")
print(f"总视频数: {total}")
print(f"违规视频: {violated}")
print(f"违规率: {vr:.1%}")
EOF
else
    echo "❌ 评估失败"
    exit 1
fi

echo ""
echo "✅ 冒烟测试成功！可以提交完整的 array job"
echo "任务完成: $(date)"
