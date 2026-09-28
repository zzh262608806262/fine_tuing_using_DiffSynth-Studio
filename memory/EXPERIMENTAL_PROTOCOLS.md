# 实验规范与 AI 协作指南

**版本**: v1.0  
**创建日期**: 2026-08-26  
**适用范围**: fine_tuing_using_DiffSynth-Studio 项目  
**面向对象**: AI 助手与科研新手协作  

---

## 目录

1. [项目定位与约束](#项目定位与约束)
2. [文件管理规范](#文件管理规范)
3. [实验记录规范](#实验记录规范)
4. [代码开发规范](#代码开发规范)
5. [作业提交规范](#作业提交规范)
6. [AI 协作原则](#ai-协作原则)
7. [学长项目经验总结](#学长项目经验总结)
8. [常见问题与解决方案](#常见问题与解决方案)

---

## 项目定位与约束

### 核心定位
本项目是基于 **DiffSynth-Studio** 开源框架进行视频生成模型微调与安全对齐研究的实验项目。

### 严格约束

#### 1. 文件修改边界（最高优先级）
```
✅ 允许修改的目录：
/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/

❌ 禁止修改的目录：
/home/x_jiage/jiage/video-unlearning/          # 学长项目，只读参考
/home/x_jiage/jiage/DiffSynth-Studio/          # 原始开源项目，只读
/home/x_jiage/.cache/                           # 系统缓存
/scratch/                                        # 集群临时存储（可能被清空）
```

**AI 协作原则**：
- 任何新增或修改的文件**必须**落在 FT 项目目录内
- 引用外部资源只能**只读访问**
- 发现文件位置异常时，AI **必须主动提醒**用户

#### 2. 集群存储策略
- **代码与小文档**：存储在 `/home5`（持久化）
- **模型与大文件**：存储在 `/scratch`（临时，可能被清空）
- **重要产物备份**：必须定期复制到 `/home5` 或项目目录

#### 3. 依赖项目关系
```
video-unlearning/  ←── 只读参考（学长成熟项目）
       ↓ 提供
   - 擦除方法（GradAscent、GradDiff 等）
   - 评测工具（NudeNet、Qwen-VL 等）
   - 项目结构范式
       ↓
fine_tuing_using_DiffSynth-Studio/  ←── 当前工作项目
   - 使用 VU 的方法和工具（通过路径引用）
   - 不修改 VU 任何文件
   - 独立的实验记录和产物
```

---

## 文件管理规范

### 1. Memory 系统（核心）

参考 video-unlearning 的 memory 组织，本项目采用以下文件：

```
memory/
├── index.md                    # 文件索引（必须维护）
├── project_details.md          # 项目技术详情
├── experiments.md              # 实验记录（编号、配置、Job ID、结果）
├── EXPERIMENTAL_PROTOCOLS.md   # 本文件
├── errors.md                   # 踩坑记录
├── learnings.md                # 学习记录
├── notes.md                    # 补充笔记
├── todo.md                     # 任务清单（仅用户变更状态）
└── feedback.md                 # 用户反馈与沟通记录
```

### 2. 文件索引维护（index.md）

**格式规范**（照搬 VU 项目）：
```markdown
| 文件 | 说明 |
|------|------|
| scripts/monitor_exp012.sh | Exp012 自动化监控脚本 |
| slurm/wan5b_bridge_verify.sbatch | J1 桥转换验证作业 |
```

**维护规则**：
- 新增/删除/重命名受索引文件时，**立即同步更新** index.md
- 路径必须**相对项目根目录**
- 说明只描述**当前职责**，不记录临时状态或实验结果
- AI 在创建新文件后**必须**更新索引

### 3. 实验产物组织

```
项目根目录/
├── models/                     # 模型权重
│   ├── Wan-AI/
│   │   └── Wan2.2-TI2V-5B-Diffusers/  # 原始模型
│   ├── unlearn/                # 擦除产物
│   └── train/                  # 微调产物
├── data/                       # 数据集
│   └── wan5b/
│       ├── unlearn_baseline/   # 基线视频
│       └── unlearn_latents/    # 缓存的 latent
├── outputs/                    # 实验输出
│   ├── exp012_monitor.log      # 监控日志
│   ├── exp012_status.json      # 实时状态
│   └── wan5b_eval/             # 评估结果
└── slurm/
    └── logs/                   # SLURM 作业日志
```

---

## 实验记录规范

### 1. 实验编号体系

参考 VU 项目的严格编号规则：

```
ExpNNN（三位数字，从 Exp001 开始）
- 编号连续递增，不重复
- 失败或废弃的实验保留编号
- 不创建空的实验目录
```

### 2. experiments.md 记录格式

**标准格式**（照搬 VU 项目）：
```markdown
## ExpNNN - 实验标题（一句话概括目的）

- **日期**: YYYY-MM-DD
- **目的**: 详细描述研究问题
- **方法**: 使用的方法、模型、超参数
- **数据**: 数据集、样本数、来源
- **配置**: 关键配置参数
  ```yaml
  key: value
  ```
- **作业 ID**:
  - J0 download: 17377125
  - J1 verify: 17377126 (node085, 历时 XX:XX)
  - J1c baseline: 17377127
  - ...
- **产物路径**: 
  - 模型: `models/xxx/`
  - 结果: `outputs/xxx/`
- **状态**: 运行中 / 完成 / 失败 / 废弃
- **结果**: 
  - 指标: accuracy=0.95
  - 发现: ...
  - 结论: ...
- **失败原因**（如果失败）: 详细记录错误信息、根因、修复方案
- **后续**: 下一步计划或衍生实验
```

### 3. 多次提交记录

当一个实验需要多次调试时：

```markdown
## Exp012 - 擦除后微调安全回潮研究

- **第一次提交** (YYYY-MM-DD HH:MM): J0=123456 → J1=123457 → ...
  - 失败原因: ...
  - 修复方案: ...
  
- **第二次提交** (YYYY-MM-DD HH:MM): J0=123458 → ...
  - 失败原因: ...
  - 修复方案: ...
  
- **第N次提交** (YYYY-MM-DD HH:MM): 最终成功
  - 全链通过
  - 结果: ...
```

### 4. 实验快照日期

涉及**当前状态**的描述必须标注日期：
```markdown
## 当前实现快照（2026-08-26）
- 已修复问题: ...
- 运行中的作业: ...
```

---

## 代码开发规范

### 1. 脚本命名与组织

**SLURM 脚本**：
```bash
slurm/
├── expNNN_submit.sh           # 全链提交脚本
├── wan5b_download.sbatch      # 模块化的子任务
├── wan5b_bridge_verify.sbatch
└── logs/
    ├── expNNN_pipeline_jobs.tsv   # 作业 ID 记录
    └── wan5b_*-<job_id>.{out,err} # 作业日志
```

**Python 脚本**：
```python
scripts/
├── monitor_exp012.sh          # 监控脚本
├── verify_bridge_equiv.py     # 验证脚本
└── generate_wan5b_eval.py     # 生成脚本
```

### 2. 环境管理

**Conda 环境**：
```bash
# 激活环境的标准方式
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

# 设置正确的库路径
export PATH="/home/x_jiage/.conda/envs/diffsynth/bin:$PATH"
export LD_LIBRARY_PATH="/home/x_jiage/.conda/envs/diffsynth/lib:$LD_LIBRARY_PATH"

# 环境验证（必须）
python -c "import pandas; print('pandas:', pandas.__version__)"
```

**避免常见错误**：
- ❌ 不要设置 `PYTHONNOUSERSITE=1`（会屏蔽 conda 包）
- ❌ 不要使用相对路径的软链接
- ✅ 使用绝对路径
- ✅ 在 sbatch 开头验证关键依赖

### 3. 依赖安装

**原则**：
- 优先使用 `conda install`（解决 C++ 库依赖）
- 必要时使用 `pip install`
- 安装后**必须验证**导入成功

```bash
# 正确示例
conda install pandas libstdcxx-ng -y
python -c "import pandas; print(pandas.__version__)"
```

### 4. 错误处理

**退出码约定**：
```bash
# 成功
exit 0

# 失败（自动停止依赖链）
echo "[ERROR] ..." >&2
exit 1
```

**日志重定向**：
```bash
#SBATCH --output=slurm/logs/job_name-%j.out
#SBATCH --error=slurm/logs/job_name-%j.err
```

---

## 作业提交规范

### 1. 依赖链设计

**使用 afterok 依赖**（任一步失败自动停链）：
```bash
J0=$(sbatch --parsable job0.sbatch)
J1=$(sbatch --parsable --dependency=afterok:$J0 job1.sbatch)
J2=$(sbatch --parsable --dependency=afterok:$J1 job2.sbatch)
```

### 2. 作业 ID 记录

**pipeline_jobs.tsv 格式**：
```tsv
stage	job_id
J0	17377125
J1	17377126
J1c	17377127
J2	17377128
```

### 3. 监控机制

**自动化监控脚本**（参考 monitor_exp012.sh）：
```bash
# 功能
- 每 60 秒检查作业状态
- 记录到 monitor.log
- 生成 JSON 状态文件
- 生成 Markdown 报告
- 异常自动告警

# 使用
bash scripts/monitor_exp012.sh monitor   # 持续监控
bash scripts/monitor_exp012.sh check     # 一次性检查
bash scripts/monitor_exp012.sh report    # 生成报告
```

### 4. 作业取消

```bash
# 取消单个作业
scancel <job_id>

# 取消依赖链（必须逐个取消）
scancel 17377125 17377126 17377127 ...
```

---

## AI 协作原则

### 1. AI 的职责

**必须做的**：
1. **严格遵守文件修改边界**（最高优先级）
2. **主动验证**环境、依赖、路径
3. **完整记录**每次实验的配置、Job ID、失败原因
4. **同步更新** index.md 文件索引
5. **持续监控**作业执行状态
6. **详细解释**技术决策和修复方案
7. **提供完整代码**，不使用占位符

**禁止做的**：
1. ❌ 修改 video-unlearning 目录的任何文件
2. ❌ 使用"稍后填写"、"待补充"等占位符
3. ❌ 假设用户具备专业知识
4. ❌ 省略错误处理和验证步骤
5. ❌ 创建文档类文件（除非用户明确要求）
6. ❌ 自动提交作业（必须经用户确认）

### 2. 沟通风格

**面向科研新手**：
- 使用清晰、简洁的语言
- 解释技术术语
- 提供完整的命令示例
- 说明每个步骤的目的

**技术决策透明化**：
```markdown
# 示例
**决策**: 将 WIDTH 从 720 改为 736

**原因**: 
- 新版 VAE spatial_scale=16
- transformer patch_size=2
- 要求像素宽是 16×2=32 的倍数
- 720/16=45（奇数）→ patchify 截断为 44
- 736/16=46（偶数）→ 正确

**证据**: 
- 错误日志: RuntimeError: tensor a (45) vs tensor b (44)
- VAE 配置: models/.../vae/config.json
```

### 3. 迭代调试流程

```
发现问题 → 诊断根因 → 设计修复 → 实施修复 → 验证 → 记录
    ↓                                                    ↑
    └────────────── 失败则重复 ───────────────────────┘
```

**关键点**：
- 每次失败都必须**诊断根因**，不能盲目重试
- 同一问题失败**2次**后，必须**换思路**
- 修复后必须**验证**（测试脚本、单元测试、smoke test）
- 所有失败原因和修复方案必须**记录到 experiments.md**

### 4. 自主性与确认

**AI 可以自主执行的**：
- 读取文件、搜索代码
- 诊断错误、提出方案
- 生成脚本、修改配置
- 监控作业状态
- 更新实验记录

**必须经用户确认的**：
- 提交 SLURM 作业
- 删除重要文件
- 修改生产环境
- 安装系统级依赖
- 重大架构调整

---

## 学长项目经验总结

### 1. video-unlearning 项目的优秀实践

#### Memory 系统
- **index.md**: 文件索引，新增文件立即更新
- **project_details.md**: 技术详情，带日期的状态快照
- **experiments.md**: 实验记录，编号连续、格式统一
- **errors.md**: 踩坑记录，避免重复错误
- **todo.md**: 任务清单，仅用户变更状态

#### 实验管理
- **严格编号体系**: ExpNNN 三位数字，连续递增
- **详细记录配置**: 超参数、数据集、作业 ID 全部记录
- **产物路径规范**: 统一的目录结构，便于追溯
- **失败记录完整**: 错误信息、根因、修复方案

#### 代码规范
- **模块化设计**: 单一职责，便于复用和测试
- **配置驱动**: 使用 Hydra 管理配置
- **Registry 模式**: 字符串键→类，动态加载
- **Source Control**: 保留官方源码作为参考（只读）

#### 质量保证
- **Source Fidelity**: 与官方实现数值对齐验证
- **单元测试**: 关键模块有完整测试
- **Visual Audit**: 生成结果可视化检查
- **文档完善**: 方法文档、审计报告、用户反馈

### 2. 可复用的模式

#### 依赖链提交
```bash
#!/bin/bash
# exp012_submit.sh
set -euo pipefail

J0=$(sbatch --parsable slurm/wan5b_download.sbatch)
J1=$(sbatch --parsable --dependency=afterok:$J0 slurm/wan5b_bridge_verify.sbatch)
J1c=$(sbatch --parsable --dependency=afterok:$J1 slurm/wan5b_baseline_regen.sbatch)
# ...

# 记录 Job ID
echo -e "stage\tjob_id" > slurm/logs/exp012_pipeline_jobs.tsv
echo -e "J0\t$J0" >> slurm/logs/exp012_pipeline_jobs.tsv
echo -e "J1\t$J1" >> slurm/logs/exp012_pipeline_jobs.tsv
```

#### 环境验证闸门
```bash
# sbatch 脚本开头
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffsynth

echo "=== 环境验证 ==="
which python
python --version
python -c "import pandas; print('pandas:', pandas.__version__)" || { echo "ERROR: pandas 未找到！" >&2; exit 1; }
```

#### 监控守护进程
```bash
# 后台运行，持续监控
nohup bash scripts/monitor_exp012.sh monitor > outputs/monitor_daemon.log 2>&1 &

# 查看进程
ps aux | grep "monitor_exp012.sh monitor" | grep -v grep
```

---

## 常见问题与解决方案

### 1. 环境依赖问题

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| `ModuleNotFoundError: pandas` | conda 环境未激活或 PYTHONNOUSERSITE=1 屏蔽 | 移除 PYTHONNOUSERSITE，显式设置 PATH |
| `GLIBCXX_3.4.29 not found` | 系统 libstdc++ 版本过旧 | 设置 `LD_LIBRARY_PATH` 指向 conda lib |
| `ImportError: FFMPEG backend` | imageio 插件问题 | 不指定插件让其自动选择，或用 ffmpeg |

### 2. 文件路径问题

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| 软链接断裂 | 使用相对路径 | 改用绝对路径 `$PWD/...` |
| 文件不存在 | /scratch 被清空 | 重要文件备份到 /home5 |
| 权限错误 | 跨用户访问 | 检查文件所有者和权限 |

### 3. 作业执行问题

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| 依赖链中断 | 前置作业失败 | 使用 afterok 依赖，检查失败原因 |
| 作业超时 | 资源不足或死循环 | 设置合理的 time limit，检查代码逻辑 |
| OOM 错误 | 内存不足 | 增加 mem 请求或优化代码 |

### 4. 数据格式问题

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| 视频形状不匹配 | 分辨率不满足倍数约束 | 检查 VAE scale_factor 和 patch_size |
| Latent 尺寸错误 | WIDTH/HEIGHT 不正确 | 确保是 scale_factor × patch_size 的倍数 |
| 扩展名识别失败 | 使用了非标准扩展名 | 保留标准扩展名（.mp4, .safetensors） |

---

## 快速检查清单

### 开始新实验前

- [ ] 确认实验编号（ExpNNN）
- [ ] 在 experiments.md 创建实验记录
- [ ] 验证环境依赖（conda activate + 导入测试）
- [ ] 检查模型和数据路径
- [ ] 准备监控脚本

### 提交作业前

- [ ] 代码通过本地测试
- [ ] sbatch 脚本有环境验证
- [ ] 设置了正确的依赖关系
- [ ] 日志路径正确
- [ ] 记录了 Job ID 到 tsv 文件

### 作业失败后

- [ ] 检查 .err 日志
- [ ] 诊断根因（不是表象）
- [ ] 设计修复方案
- [ ] 记录失败原因到 experiments.md
- [ ] 验证修复后再重新提交

### 实验完成后

- [ ] 验证产物完整性
- [ ] 记录结果到 experiments.md
- [ ] 更新 index.md（如有新文件）
- [ ] 生成最终报告
- [ ] 备份重要产物到 /home5

---

## 附录：项目目录结构

```
fine_tuing_using_DiffSynth-Studio/
├── README.md
├── memory/
│   ├── index.md
│   ├── project_details.md
│   ├── experiments.md
│   ├── EXPERIMENTAL_PROTOCOLS.md  # 本文件
│   ├── errors.md
│   ├── learnings.md
│   ├── notes.md
│   ├── todo.md
│   └── feedback.md
├── specs/
│   └── unlearn-then-finetune-wan5b/
│       ├── spec.md
│       ├── tasks.md
│       └── checklist.md
├── docs/
│   └── monitoring_guide.md
├── scripts/
│   ├── monitor_exp012.sh
│   ├── verify_bridge_equiv.py
│   ├── verify_bridge_native.py
│   ├── run_unlearn_wan5b.py
│   ├── regen_unlearn_baseline_wan5b.py
│   └── generate_wan5b_eval.py
├── slurm/
│   ├── exp012_submit.sh
│   ├── wan5b_download.sbatch
│   ├── wan5b_bridge_verify.sbatch
│   ├── wan5b_baseline_regen.sbatch
│   ├── wan5b_unlearn.sbatch
│   ├── wan5b_merge_bridge.sbatch
│   ├── wan5b_finetune.sbatch
│   ├── wan5b_eval_gen.sbatch
│   ├── wan5b_eval_judge.sbatch
│   └── logs/
│       ├── exp012_pipeline_jobs.tsv
│       └── wan5b_*-<job_id>.{out,err}
├── models/
│   ├── Wan-AI/
│   ├── DiffSynth-Studio/
│   ├── unlearn/
│   └── train/
├── data/
│   └── wan5b/
├── outputs/
│   ├── exp012_monitor.log
│   ├── exp012_status.json
│   ├── exp012_report.md
│   └── wan5b_eval/
└── diffsynth/  # DiffSynth-Studio 本地副本
```

---

## 版本历史

- **v1.0** (2026-08-26): 初版，基于 video-unlearning 项目经验和 Exp012 的14次迭代教训

---

**重要提醒**：本文档是给 AI 助手的指导文件，旨在确保 AI 能够准确理解实验要求和操作标准。请在每次启动新任务时向 AI 提供本文档，并要求 AI 严格遵守其中的规范。
