# GitHub & HuggingFace 迁移计划

**目标**: 将项目迁移到新集群，代码/记录提交 GitHub，数据/checkpoint 提交 HuggingFace

**日期**: 2026-09-22  
**仓库**: 
- GitHub: https://github.com/zzh262608806262/fine_tuing_using_DiffSynth-Studio
- HuggingFace: (待创建)

---

## 📋 目录

1. [当前状态检查](#1-当前状态检查)
2. [GitHub 提交清单](#2-github-提交清单)
3. [HuggingFace 提交清单](#3-huggingface-提交清单)
4. [.gitignore 审查](#4-gitignore-审查)
5. [提交步骤](#5-提交步骤)

---

## 1. 当前状态检查

### 1.1 Git 状态
```bash
# 总共 345 个未跟踪/修改的文件
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
git status --short | wc -l
# 输出: 345
```

### 1.2 已 ignore 的大文件目录
```
✅ /data          - 数据集（应提交 HF）
✅ /models        - 模型 checkpoint（应提交 HF）
✅ /outputs       - 生成的视频/评估结果（应提交 HF）
✅ /slurm/logs    - SLURM 日志（不提交）
✅ *.safetensors  - 模型权重（应提交 HF）
✅ *.pth, *.pt    - PyTorch 权重（应提交 HF）
```

---

## 2. GitHub 提交清单

### 2.1 ✅ 应该提交的文件（代码 + 文档）

#### 核心代码
- ✅ `scripts/*.py` - 所有实验脚本
- ✅ `slurm/*.sbatch` - SLURM 作业脚本
- ✅ `slurm/*.sh` - 提交脚本
- ✅ `diffsynth/` - 修改后的 DiffSynth 库
- ✅ `examples/` - 训练/推理示例
- ✅ `classify/` - 分类器相关代码

#### 项目文档
- ✅ `memory/*.md` - 所有实验记录和项目文档
  - ✅ `experiments.md` - 完整实验记录
  - ✅ `MASTER_SUMMARY.md` - 项目总览
  - ✅ `QUICK_DATA_REFERENCE.md` - 数据快速查询
  - ✅ `group_meeting_presentation.md` - 组会汇报（新增）
  - ✅ `project_details.md` - 项目技术细节
  - ✅ `EXPERIMENTAL_PROTOCOLS.md` - 实验规范
  - ✅ `errors.md` - 错误记录
  - ✅ 其他 memory 文件
- ✅ `docs/*.md` - 实验报告和文档
- ✅ `.claude/CLAUDE.md` - AI 协作指南

#### 配置文件
- ✅ `requirements.txt` / `environment.yml` - 依赖配置
- ✅ `README.md` - 项目说明
- ✅ `.gitignore` - Git 忽略规则

#### 小型数据文件（如果存在）
- ✅ `data/tiger_dataset/metadata_*.csv` - 数据集 metadata（CSV 小文件）
- ✅ `classify/configs/*.json` - 配置文件

### 2.2 ❌ 不应提交到 GitHub 的文件

#### 临时文档（项目根目录）
- ❌ `EXP019_*.md` - 临时调试文档（共 9 个）
- ❌ `DISTILL_*.md` - 临时文档
- ❌ `READY_TO_SUBMIT.md`, `QUICK_START_DISTILL.md` 等
- ❌ `check_exp017_v28.sh` - 临时调试脚本

**建议**: 清理或移到 `docs/archive/` 或 `memory/temp/`

#### AI 工具配置
- ❌ `.claude/` - Claude Code 配置（个人环境特定）

#### 日志和缓存
- ❌ `slurm/logs/` - SLURM 日志（已 ignore）
- ❌ `__pycache__/` - Python 缓存（已 ignore）

#### 大文件
- ❌ `/models` - 模型权重（→ HuggingFace）
- ❌ `/data` - 数据集（→ HuggingFace）
- ❌ `/outputs` - 生成结果（→ HuggingFace）

---

## 3. HuggingFace 提交清单

### 3.1 数据集（Datasets Repository）

建议创建独立的 HF Dataset: `littlepig404/wan5b-unlearning-experiments`

#### 结构
```
wan5b-unlearning-experiments/
├── datasets/
│   ├── tiger_dataset/          # 微调/蒸馏训练数据
│   │   ├── metadata_100.csv
│   │   ├── metadata_100_distill.csv
│   │   └── videos/*.mp4        # 100 个视频
│   └── vu_benchmark/           # VU 评估数据（如果允许）
│       └── benchmark_wan.jsonl
├── models/
│   ├── unlearned/              # 擦除后的 LoRA adapters
│   │   ├── exp015_esd/
│   │   │   └── step-600.safetensors  (~20MB)
│   │   ├── exp015_npo/
│   │   ├── exp015_grad_ascent/
│   │   ├── exp015_anchor_distill/
│   │   ├── exp019_graddiff/
│   │   └── exp020a_npo_masked/
│   ├── finetune/               # 微调后的 LoRA
│   │   ├── exp018_esd_ft/
│   │   │   ├── epoch-1.safetensors
│   │   │   ├── ...
│   │   │   └── epoch-20.safetensors
│   │   └── ... (其他方法)
│   ├── quantized/              # 量化模型（如果大小允许）
│   │   └── exp016_*/
│   └── distilled/              # 蒸馏模型
│       └── exp017_*_distill/
└── outputs/
    ├── videos/                 # 生成的视频（选择性提交）
    │   ├── exp015/
    │   ├── exp016/
    │   ├── exp017/
    │   └── exp018/
    └── evaluations/            # 评估结果 JSON
        ├── exp015_evaluation/
        ├── exp016_evaluation/
        ├── exp017_evaluation/
        └── exp018_evaluation/
```

**大小估算**:
- LoRA adapters: ~20MB × 30 个 checkpoint ≈ 600MB
- 评估 JSON: ~100KB × 50 个 ≈ 5MB
- 视频（可选）: ~5MB × 2000 个 ≈ 10GB

### 3.2 模型（Model Repository）

如果需要独立的模型仓库，可以创建：
- `littlepig404/wan5b-esd-erased` - ESD 擦除后模型
- `littlepig404/wan5b-graddiff-erased` - GradDiff 擦除后模型
- 等等...

**注意**: 完整合并后的模型 ~10GB，考虑是否需要上传全部

---

## 4. .gitignore 审查

### 4.1 当前 .gitignore 关键规则

```gitignore
# ✅ 正确 - 大文件/数据
/data
/models
/outputs
/slurm/logs/

# ✅ 正确 - 模型文件扩展名
*.safetensors
*.pth
*.ckpt
*.pt
*.bin

# ✅ 正确 - Python 标准
__pycache__/
*.pyc
.venv/

# ✅ 正确 - 特殊规则
data/malicious_dataset/          # Exp008 投毒数据
models/train/*lora_malicious*    # 恶意模型
```

### 4.2 建议新增规则

```gitignore
# 临时调试文档
EXP019_*.md
DISTILL_*.md
READY_TO_SUBMIT.md
QUICK_START_*.md
SMOKE_TEST_GUIDE.md
check_exp*.sh

# Claude Code 配置（个人环境特定）
.claude/

# 备份文件
*_BACKUP/
*.bak
*.backup

# 大型日志
*.log
slurm-*.out
```

---

## 5. 提交步骤

### 5.1 清理临时文件

```bash
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 1. 移动临时文档到 archive
mkdir -p docs/archive
mv EXP019_*.md DISTILL_*.md READY_TO_SUBMIT.md QUICK_START_*.md SMOKE_TEST_GUIDE.md docs/archive/
mv check_exp017_v28.sh docs/archive/

# 2. 更新 .gitignore
echo "" >> .gitignore
echo "# 临时调试文档" >> .gitignore
echo "docs/archive/" >> .gitignore
echo "" >> .gitignore
echo "# Claude Code 配置" >> .gitignore
echo ".claude/" >> .gitignore
```

### 5.2 GitHub 提交

```bash
# 1. 检查远程仓库
git remote -v
# 如果没有，添加：
# git remote add origin https://github.com/zzh262608806262/fine_tuing_using_DiffSynth-Studio.git

# 2. 创建新分支（安全起见）
git checkout -b migration-cleanup

# 3. 添加所有代码和文档
git add scripts/ slurm/ diffsynth/ examples/ classify/
git add memory/ docs/
git add .gitignore README.md requirements.txt

# 4. 查看将要提交的文件
git status

# 5. 提交
git commit -m "feat: add all experiments (Exp015-020) and documentation

- Add unlearning methods: ESD, NPO, GradAscent, AnchorDistill, GradDiff
- Add fine-tuning, quantization, distillation experiments
- Add comprehensive experiment records and project documentation
- Add group meeting presentation materials

Co-Authored-By: Claude Code <noreply@anthropic.com>"

# 6. 推送
git push origin migration-cleanup

# 7. 合并到 main（确认无误后）
git checkout main
git merge migration-cleanup
git push origin main
```

### 5.3 HuggingFace 提交

```bash
# 1. 安装 HF CLI
pip install huggingface_hub

# 2. 登录
huggingface-cli login

# 3. 创建 Dataset Repository
huggingface-cli repo create wan5b-unlearning-experiments --type dataset

# 4. 克隆仓库
cd /tmp
git clone https://huggingface.co/datasets/littlepig404/wan5b-unlearning-experiments
cd wan5b-unlearning-experiments

# 5. 使用 git-lfs（大文件支持）
git lfs install
git lfs track "*.safetensors"
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.mp4"

# 6. 复制文件
# LoRA adapters
cp -r /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/models/unlearn models/
cp -r /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/models/finetune models/

# 评估结果
mkdir -p outputs/evaluations
cp -r /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/outputs/exp*/evaluation outputs/evaluations/

# 小型数据
cp -r /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/data/tiger_dataset datasets/

# 7. 创建 README
cat > README.md << 'EOF'
# Wan5B Unlearning Experiments Dataset

This dataset contains models, evaluation results, and training data for video unlearning experiments on Wan2.2-TI2V-5B.

## Contents

- `models/unlearned/`: LoRA adapters for erased models (~20MB each)
- `models/finetune/`: Fine-tuned LoRA checkpoints
- `outputs/evaluations/`: NudeNet evaluation results (JSON)
- `datasets/tiger_dataset/`: Training data for fine-tuning/distillation

## Experiments

- **Exp015**: 4 unlearning methods (ESD, NPO, GradAscent, AnchorDistill) + fine-tuning
- **Exp016**: Quantization (NF4) impact on erased models
- **Exp017**: Distillation (50→4 steps) impact
- **Exp018**: High-density fine-tuning (500 exposures, 20 checkpoints)
- **Exp019**: GradDiff (with retain constraint)
- **Exp020a**: NPOMasked (with attention mask)

## Usage

See [GitHub repository](https://github.com/zzh262608806262/fine_tuing_using_DiffSynth-Studio) for full code and documentation.
EOF

# 8. 提交
git add .
git commit -m "Initial dataset: LoRA adapters, evaluations, and training data"
git push
```

---

## 6. 检查清单

### 提交前检查

- [ ] 清理临时文档（EXP019_*.md 等）
- [ ] 更新 .gitignore（添加 .claude/, docs/archive/）
- [ ] 确认不包含敏感信息（绝对路径、用户名等）
- [ ] 确认 README.md 完整（项目说明、使用方法）
- [ ] 确认 requirements.txt 或 environment.yml 存在

### GitHub 提交后检查

- [ ] 所有代码文件可见
- [ ] memory/*.md 文档完整
- [ ] .gitignore 生效（models/, data/, outputs/ 未提交）
- [ ] CLAUDE.md 可见（给新用户提供 AI 协作指南）

### HuggingFace 提交后检查

- [ ] LoRA adapters 可下载
- [ ] 评估 JSON 完整
- [ ] README 清晰（说明文件结构和用法）
- [ ] git-lfs 配置正确（大文件未损坏）

---

## 7. 新集群部署步骤

### 7.1 从 GitHub 克隆代码

```bash
# 新集群上
git clone https://github.com/zzh262608806262/fine_tuing_using_DiffSynth-Studio.git
cd fine_tuing_using_DiffSynth-Studio

# 安装依赖
conda env create -f environment.yml
conda activate diffsynth
```

### 7.2 从 HuggingFace 下载数据

```bash
# 安装 HF CLI
pip install huggingface_hub

# 下载整个 dataset
huggingface-cli download littlepig404/wan5b-unlearning-experiments \
    --repo-type dataset \
    --local-dir ./hf_data

# 链接到项目结构
ln -s ./hf_data/models models
ln -s ./hf_data/datasets/tiger_dataset data/tiger_dataset
ln -s ./hf_data/outputs outputs
```

### 7.3 下载基座模型

```bash
# 下载 Wan2.2-TI2V-5B（如果 HF 上有）
mkdir -p models/Wan-AI
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir models/Wan-AI/Wan2.2-TI2V-5B

# 或手动从原始位置复制
```

### 7.4 验证环境

```bash
# 运行 smoke test
python scripts/debug/smoke_test.py

# 确认模型路径正确
ls -lh models/Wan-AI/Wan2.2-TI2V-5B
ls -lh models/unlearn/exp015_esd
```

---

## 8. 注意事项

### 8.1 敏感信息检查

❌ **不要提交**:
- 绝对路径（如 `/home/x_jiage/...`）
- 用户名/邮箱
- API keys
- 服务器 IP/域名

✅ **替换为**:
- 相对路径（如 `./models/...`）
- 环境变量（如 `$USER`）
- 占位符（如 `YOUR_API_KEY`）

### 8.2 文件大小限制

- **GitHub**: 单文件 < 100MB，仓库 < 1GB
- **HuggingFace**: 使用 git-lfs，单文件可达 50GB

### 8.3 许可证

确认项目根目录有 `LICENSE` 文件（建议 MIT 或 Apache 2.0）

### 8.4 恶意内容（Exp008）

✅ **已正确 ignore**:
```gitignore
data/malicious_dataset/
models/train/*lora_malicious*
outputs/safesora_gen/malicious/
```

**禁止上传任何 Exp008 投毒相关数据到公开仓库！**

---

## 9. 下一步

1. **立即执行**: 清理临时文件，更新 .gitignore
2. **GitHub 提交**: 先创建 `migration-cleanup` 分支测试
3. **HuggingFace 准备**: 先上传小文件（评估 JSON），验证流程
4. **逐步上传**: LoRA adapters → 评估结果 → 数据集
5. **新集群测试**: 从 GitHub + HF 完整部署，验证流程

---

**准备好后告诉我，我会帮你逐步执行！**
