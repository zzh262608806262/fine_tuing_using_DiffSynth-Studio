#!/bin/bash
# 将硬编码的绝对路径替换为环境变量或相对路径

PROJECT_ROOT="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio"
cd "$PROJECT_ROOT"

echo "=== 替换绝对路径为相对路径/环境变量 ==="

# 1. VU_ROOT 路径（指向外部 video-unlearning 项目）
# 替换为环境变量 ${VU_ROOT}，默认值为相对路径
find scripts/ -type f \( -name "*.py" -o -name "*.sh" \) -exec sed -i 's|VU_ROOT = Path(os.getenv("VU_ROOT", "../video-unlearning"))|VU_ROOT = Path(os.getenv("VU_ROOT", "../video-unlearning"))|g' {} \;
find scripts/ -type f \( -name "*.py" -o -name "*.sh" \) -exec sed -i 's|VU_ROOT = os.getenv("VU_ROOT", "../video-unlearning")|VU_ROOT = os.getenv("VU_ROOT", "../video-unlearning")|g' {} \;

# 2. PROJECT_ROOT 路径（当前项目）
# 使用 __file__ 动态获取或相对路径
find scripts/ -type f -name "*.py" -exec sed -i "s|ROOT='/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio'|ROOT=Path(__file__).parent.parent.absolute()|g" {} \;
find scripts/ -type f -name "*.py" -exec sed -i "s|FT_ROOT = Path(\"/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio\")|FT_ROOT = Path(__file__).parent.parent.absolute()|g" {} \;

# 3. 脚本中的 cd 命令
find scripts/ -type f -name "*.sh" -exec sed -i 's|cd "$(dirname "$(dirname "$(readlink -f "$0")")")" # Auto: project root|cd "$(dirname "$(dirname "$(readlink -f "$0")")")" # Auto: project root|g' {} \;

# 4. LOG_FILE 路径
find scripts/ -type f -name "*.sh" -exec sed -i 's|LOG_FILE="slurm/logs/|LOG_FILE="slurm/logs/|g' {} \;

# 5. 模型路径（Qwen3-VL）
find scripts/ -type f -name "*.py" -exec sed -i 's|MODEL_PATH = "/home/x_jiage/jiage/models/Qwen3-VL-8B-Instruct"|MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "../models/Qwen3-VL-8B-Instruct")|g' {} \;

echo "=== 检查替换结果 ==="
echo "剩余的绝对路径："
grep -r "/home/x_jiage" scripts/ slurm/ --include="*.py" --include="*.sh" | wc -l

echo ""
echo "详细列表："
grep -r "/home/x_jiage" scripts/ slurm/ --include="*.py" --include="*.sh" | head -10
