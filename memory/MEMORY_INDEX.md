# Memory 文件索引

本目录包含项目的核心文档和实验记录。

## 📋 核心文档

### 实验相关
- **experiments.md** - 完整实验记录（Exp001-Exp020）
  - 所有实验的配置、脚本、结果、产物
  - 包含错误记录和修复方案
  
- **EXPERIMENTAL_PROTOCOLS.md** - 实验规范
  - 标准化流程和命名规范
  - 数据收集和评估协议

- **QUICK_DATA_REFERENCE.md** - 数据快速查询
  - 实验结果速查表
  - 违规率数据汇总

### 项目管理
- **MASTER_SUMMARY.md** - 项目总览
  - 项目背景和目标
  - 技术栈和依赖
  - 目录结构说明

- **group_meeting_presentation.md** - 组会汇报材料
  - 研究背景与动机
  - VU 擦除方法技术详解
  - 实验结果汇总
  - 核心发现与讨论

### 迁移文档
- **github_huggingface_migration_plan.md** - 迁移计划
  - GitHub 提交清单
  - HuggingFace 提交清单
  - 迁移步骤

- **github_integrity_report.md** - GitHub 完整性报告
  - 代码完整性验证
  - 文件清单和大小统计

- **hf_upload_strategy_final.md** - HF 上传策略
  - 精选上传内容
  - 数据量分析

## 🔍 如何使用

### 快速查找实验结果
```bash
# 查看特定实验
grep -A 50 "## Exp 018" experiments.md

# 查看违规率数据
less QUICK_DATA_REFERENCE.md
```

### 复现实验
1. 阅读 `experiments.md` 中的实验配置
2. 查看对应的脚本文件（`scripts/` 和 `slurm/`）
3. 从 HuggingFace 下载必需的模型和数据
4. 运行脚本

### 理解项目架构
1. 阅读 `MASTER_SUMMARY.md` 了解整体结构
2. 阅读 `group_meeting_presentation.md` 了解技术细节
3. 查看 `EXPERIMENTAL_PROTOCOLS.md` 了解规范

## 📝 其他文档

项目根目录的其他重要文档：
- `README.md` - 项目介绍
- `NEW_CLUSTER_SETUP.md` - 新集群部署指南
- `MIGRATION.md` - 迁移指南（环境变量配置）
- `env.template` - 环境变量模板

## 🤖 Claude Code

如果使用 Claude Code，它会自动加载这些文档作为上下文。

你可以问：
- "Exp018 的违规率是多少？"
- "如何运行 ESD 擦除实验？"
- "GradDiff 为什么失效？"

Claude 会从这些文档中查找答案。
