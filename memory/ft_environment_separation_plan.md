# 创建独立的 FT 环境方案

## 背景

当前问题：`diffsynth` 环境被频繁升级且与多个项目共享，导致版本冲突

## 解决方案：创建独立环境

### 方案A：基于当前环境克隆（推荐）

```bash
# 1. 克隆现有环境作为基础
conda create --name ft_diffsynth --clone diffsynth

# 2. 激活新环境
conda activate ft_diffsynth

# 3. 降级关键包到稳定版本
pip install numpy==1.26.4  # numpy 1.x 最后稳定版
pip install 'pandas>=2.0,<2.1'
pip install transformers==4.44.0
pip install 'tokenizers>=0.19,<0.20'

# 4. 验证环境
python /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/scripts/test_env.py

# 5. 固定版本
pip freeze > /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/requirements_ft.txt
```

### 方案B：从头创建（更干净但耗时）

```bash
# 1. 创建新环境
conda create -n ft_diffsynth python=3.10

# 2. 安装核心包
conda activate ft_diffsynth
pip install torch==2.4.0 torchvision==0.19.0
pip install transformers==4.44.0 accelerate diffusers
pip install numpy==1.26.4 pandas==2.0.3
pip install opencv-python pillow imageio

# 3. 安装 DiffSynth Studio 依赖
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
pip install -e .

# 4. 验证并固定
python scripts/test_env.py
pip freeze > requirements_ft.txt
```

---

## 修改所有 FT 项目脚本使用新环境

### 1. 修改 sbatch 脚本

所有 `slurm/*.sbatch` 文件中：

```bash
# 旧的
conda activate diffsynth

# 改为
conda activate ft_diffsynth
```

### 2. 修改交互式脚本

`scripts/exp017_interactive_test.sh` 等：

```bash
# 旧的
conda activate diffsynth

# 改为
conda activate ft_diffsynth
```

### 3. 使用 VU 脚本时

当调用 VU 项目的评估脚本时：

```bash
# 显式激活 VU 环境
conda activate diffsynth  # 或 VU 项目的环境名
python /home/x_jiage/jiage/video-unlearning/scripts/eval/score_nudity_rate.py ...
```

---

## 环境隔离策略

### FT 项目（ft_diffsynth）
- 蒸馏训练
- 微调训练
- 视频生成（使用 FT 训练的模型）

### VU 项目（diffsynth 或独立环境）
- 擦除方法
- 安全评估
- NudeNet 检测

### 共享原则
- **不共享**：只在各自项目使用
- **明确切换**：跨项目调用时显式切换环境

---

## 实施步骤

### 第1步：创建环境（现在）

```bash
# 选择方案A（快速）
conda create --name ft_diffsynth --clone diffsynth
conda activate ft_diffsynth

# 修复版本冲突
pip install numpy==1.26.4
pip install --force-reinstall --no-cache-dir 'pandas>=2.0,<2.1'

# 验证
python /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/scripts/test_env.py
```

### 第2步：批量修改脚本

```bash
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 备份
cp -r slurm slurm_backup_$(date +%Y%m%d)

# 批量替换（AI 辅助）
# 将所有 slurm/*.sbatch 和 scripts/*.sh 中的
# "conda activate diffsynth" 改为 "conda activate ft_diffsynth"
```

### 第3步：测试

```bash
# 在登录节点测试环境
conda activate ft_diffsynth
python scripts/test_env.py

# 如果通过，重新提交 exp017
bash slurm/exp017_submit_all.sh
```

---

## 优点

1. ✅ **隔离性**：FT 环境升级不影响 VU，反之亦然
2. ✅ **稳定性**：版本固定在 requirements_ft.txt
3. ✅ **可重现**：任何时候都能从 requirements 重建
4. ✅ **调试友好**：问题范围缩小到单个项目

---

## 注意事项

1. **磁盘空间**：克隆环境会占用额外 ~10GB
2. **记忆负担**：需要记住用哪个环境
3. **文档更新**：更新 `memory/project_details.md` 记录环境名

---

**建议立即执行方案A**，然后批量修改脚本使用 `ft_diffsynth` 环境。
