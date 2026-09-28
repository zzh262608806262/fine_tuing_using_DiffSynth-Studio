# Memory 目录说明

本目录存放项目的持久化记录文件，供 AI 对话使用。

## 📖 新对话建议阅读顺序

1. **MASTER_SUMMARY.md** - 项目总览（5分钟了解全貌）
2. **QUICK_DATA_REFERENCE.md** - 数据查询手册（快速查数据）
3. **experiments.md** - 完整实验记录（需要细节时查阅）

## 📂 文件分类

### 核心文档
- `MASTER_SUMMARY.md` - 项目总览与快速索引
- `QUICK_DATA_REFERENCE.md` - 实验数据快速查询手册
- `experiments.md` - 完整实验记录（Exp001-Exp018）
- `index.md` - 所有文件索引
- `project_details.md` - 项目技术细节

### 实验规范
- `EXPERIMENTAL_PROTOCOLS.md` - 实验协议与规范

### 专项记录
- `exp015_exp018_baseline_data.md` - Exp015/018完整数据表
- `exp009_multi_judge_comparison.md` - 多判别器对比实验
- `exp010_safesora_test_multi_judge.md` - SafeSora测试集评估
- `safesora_label_audit.md` - SafeSora标注可信度审计

### 工具与方法
- `gpu_node_apply.md` - GPU节点申请方法
- `improve_safe_binary_accuracy.md` - 分类器优化计划
- `wan5b_distillation_guide.md` - Wan5B蒸馏指南
- `script_renaming_pitfall.md` - 脚本重命名陷阱

### 错误记录
- `errors.md` - 通用踩坑记录
- `exp015_errors.md` - Exp015错误记录

### 临时状态文件（可忽略）
所有 `exp01X_*_status*.md` 文件都是临时状态，已整合到 `experiments.md`

## 🔄 维护原则

1. **experiments.md 是唯一真相来源** - 所有实验结果最终记录在这里
2. **MASTER_SUMMARY.md 保持简洁** - 只放关键发现和快速索引
3. **临时状态文件可删除** - 实验完成后状态已整合到 experiments.md
4. **新实验必须更新** - experiments.md + MASTER_SUMMARY.md + QUICK_DATA_REFERENCE.md

## 📊 当前状态 (2026-09-16)

- ✅ Exp001-Exp018 全部记录完整
- ✅ Exp015/016/018 评估结果已更新
- ⚠️ Exp017 缺评估（495个蒸馏视频待评估）
- 📊 总计 2,475 个视频，2,079 个已评估
