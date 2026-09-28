#!/bin/bash
# GitHub 代码完整性检查脚本

echo "=========================================="
echo "GitHub 代码完整性检查"
echo "=========================================="
echo ""

# 1. 检查关键目录结构
echo "1. 检查关键目录是否存在..."
DIRS=(
    "scripts"
    "slurm"
    "memory"
    "docs"
    "classify"
    "diffsynth"
    "examples"
    "data"
    "models"
)

for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f 2>/dev/null | wc -l)
        echo "  ✅ $dir/ - $count 个文件"
    else
        echo "  ❌ $dir/ - 缺失"
    fi
done

echo ""
echo "2. 检查关键文件是否存在..."
FILES=(
    "README.md"
    "MIGRATION.md"
    "env.template"
    ".gitignore"
    "memory/experiments.md"
    "memory/MASTER_SUMMARY.md"
    "memory/QUICK_DATA_REFERENCE.md"
    "memory/group_meeting_presentation.md"
    "scripts/run_unlearn_wan5b.py"
    "scripts/evaluate_videos_nudenet.py"
    "slurm/exp015_submit.sh"
    "slurm/exp018_submit_all.sh"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        echo "  ✅ $file - ${size} bytes"
    else
        echo "  ❌ $file - 缺失"
    fi
done

echo ""
echo "3. 检查 scripts/ 目录内容..."
echo "  Python 脚本: $(find scripts/ -name "*.py" 2>/dev/null | wc -l)"
echo "  Shell 脚本: $(find scripts/ -name "*.sh" 2>/dev/null | wc -l)"

echo ""
echo "4. 检查 slurm/ 目录内容..."
echo "  SLURM 脚本: $(find slurm/ -name "*.sbatch" 2>/dev/null | wc -l)"
echo "  提交脚本: $(find slurm/ -name "*.sh" 2>/dev/null | wc -l)"

echo ""
echo "5. 检查 memory/ 文档..."
echo "  实验记录: $(find memory/ -name "*.md" 2>/dev/null | wc -l)"

echo ""
echo "6. 检查是否有敏感信息（绝对路径）..."
sensitive_count=$(grep -r "/home/x_jiage" scripts/ slurm/ memory/ --include="*.py" --include="*.sh" --include="*.sbatch" 2>/dev/null | wc -l)
if [ "$sensitive_count" -gt 0 ]; then
    echo "  ⚠️  发现 $sensitive_count 处绝对路径"
    echo "  示例："
    grep -r "/home/x_jiage" scripts/ slurm/ --include="*.py" --include="*.sh" 2>/dev/null | head -3
else
    echo "  ✅ 未发现明显的绝对路径"
fi

echo ""
echo "7. 检查 .gitignore 是否生效..."
if [ -f ".gitignore" ]; then
    echo "  ✅ .gitignore 存在"
    echo "  忽略规则："
    grep -v "^#" .gitignore | grep -v "^$" | head -10
else
    echo "  ❌ .gitignore 缺失"
fi

echo ""
echo "8. 检查大文件目录是否被忽略..."
BIG_DIRS=("data" "models" "outputs")
for dir in "${BIG_DIRS[@]}"; do
    if git ls-files "$dir" 2>/dev/null | grep -q .; then
        echo "  ⚠️  $dir/ 中有文件被跟踪（可能不应该）"
        git ls-files "$dir" | head -3
    else
        echo "  ✅ $dir/ 已被正确忽略"
    fi
done

echo ""
echo "=========================================="
echo "检查完成！"
echo "=========================================="
