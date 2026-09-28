# 迁移到新集群注意事项

本项目代码中包含一些硬编码的绝对路径（主要在 `scripts/` 和 `slurm/` 目录）。这些路径在原始集群（Berzelius）上是有效的，但在新集群上需要修改。

## 快速配置方法

### 1. 设置环境变量

```bash
# 复制环境变量模板
cp env.template .env

# 编辑 .env，修改为你的集群路径
vim .env

# 主要需要修改的变量：
# - PROJECT_ROOT: 项目根目录
# - VU_ROOT: video-unlearning 项目路径
# - CONDA_ROOT: conda 安装路径
```

### 2. 在使用脚本前加载环境变量

```bash
# 方法 1: 手动加载（每次运行前）
source .env

# 方法 2: 在 ~/.bashrc 中添加（永久生效）
echo "source /path/to/fine_tuing_using_DiffSynth-Studio/.env" >> ~/.bashrc
```

### 3. 已修复的文件

以下文件已经替换为环境变量或相对路径：

- `scripts/*.py` - Python 脚本中的 VU_ROOT, PROJECT_ROOT
- `scripts/*.sh` - Shell 脚本中的项目路径

### 4. 仍需手动修改的文件

以下文件可能仍包含硬编码路径，需要根据实际情况修改：

- `slurm/*.sbatch` - SLURM 作业脚本（219 处）
- `scripts/debug/*` - 调试脚本（44 处）

**建议**: 使用全局替换命令：

```bash
# 替换项目根路径
find slurm/ -type f -name "*.sbatch" -exec sed -i 's|/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio|${PROJECT_ROOT}|g' {} \;

# 替换 VU 项目路径
find slurm/ -type f -name "*.sbatch" -exec sed -i 's|/home/x_jiage/jiage/video-unlearning|${VU_ROOT}|g' {} \;

# 替换 conda 路径
find slurm/ -type f -name "*.sbatch" -exec sed -i 's|/home/x_jiage/miniforge3|${CONDA_ROOT}|g' {} \;
```

## 路径清单

### 项目内路径（相对路径可用）
- `models/` - 模型权重和 checkpoint
- `data/` - 数据集
- `outputs/` - 生成的视频和评估结果
- `scripts/` - Python 脚本
- `slurm/` - SLURM 作业脚本

### 外部依赖路径（需要配置）
- Video Unlearning 项目: `${VU_ROOT}`
  - 默认假设在: `../video-unlearning`
  - 用于: 擦除方法、评估工具、数据集
- Conda 环境: `${CONDA_ROOT}`
  - 默认假设在: `${HOME}/miniforge3`
  - 需要: `diffsynth` 和 `vu` 两个环境

## 验证配置

```bash
# 1. 检查环境变量
source .env
echo $PROJECT_ROOT
echo $VU_ROOT

# 2. 检查必需的目录存在
ls $PROJECT_ROOT/models/Wan-AI/Wan2.2-TI2V-5B
ls $VU_ROOT/src/

# 3. 检查 conda 环境
conda env list | grep diffsynth
conda env list | grep vu

# 4. 运行简单测试
python -c "import sys; sys.path.insert(0, '$VU_ROOT'); from src.eval.detectors.nudenet import NudeNetDetector; print('OK')"
```

## 常见问题

### Q: 为什么不全部使用相对路径？
A: SLURM 脚本在不同工作目录下执行，相对路径可能失效。环境变量更稳定。

### Q: 必须保持相同的目录结构吗？
A: 不必须。只需要在 `.env` 中正确配置路径即可。

### Q: 如何知道哪些路径需要修改？
A: 运行时如果出现 `FileNotFoundError` 或 `ModuleNotFoundError`，检查错误信息中的路径。

## 联系方式

如有问题，请参考：
- 项目文档: `memory/experiments.md`
- AI 协作指南: `.claude/CLAUDE.md`
- 技术细节: `memory/project_details.md`
