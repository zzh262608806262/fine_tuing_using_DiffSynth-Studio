# Exp 017 执行状态报告

**日期**: 2026-09-03  
**时间**: 15:48  
**状态**: ✅ 已提交，正在排队

---

## 📋 提交确认

### 回答你的三个确认问题

#### 1️⃣ VU 五种擦除方法覆盖情况

**当前实现**: 2种方法（GradAscent + ESD）

| 方法 | 状态 | 说明 |
|------|------|------|
| ✅ GradAscent | 已包含 | Job 17451263 |
| ❌ GradDiff | 未支持 | run_unlearn_wan5b.py 不支持 |
| ✅ ESD | 已包含 | Job 17451264 |
| ⚠️ NPO | 未包含 | 可扩展（在 ARMS 数组添加 "npo"）|
| ❌ AnchorDistill | Exp015失败 | prompt缺nudity概念词 |

**建议**: 先验证 2 种方法概念，结果好再扩展 NPO

#### 2️⃣ 蒸馏脚本参考来源

**✅ 完全基于 Wan2.1-1.3B 蒸馏**

参考来源：
1. DiffSynth 官方: `examples/wanvideo/model_training/special/direct_distill/Wan2.1-T2V-1.3B.sh`
2. 本项目 Exp 003: `scripts/distill.py`

关键适配：
- VAE: Wan2.1 → Wan2.2 (3D VAE)
- 帧数: 81 → 17 (与擦除训练对齐)
- 宽度: 832 → 736 (32倍数约束)
- 输入: 新增 input_image (TI2V)

#### 3️⃣ 参数合理性

**✅ 参数合理**

| 参数 | 1.3B蒸馏 | Exp017 | 验证 |
|------|----------|--------|------|
| learning_rate | 1e-5 | 1e-5 | ✅ 相同 |
| num_epochs | 2 | 2 | ✅ 相同 |
| dataset_repeat | 160 | 160 | ✅ 相同 |
| num_frames | 81 | 17 | ⚠️ 对齐擦除训练 |
| width | 832 | 736 | ⚠️ Wan2.2 VAE要求 |
| task | direct_distill | direct_distill | ✅ 相同 |
| 目标步数 | 4 | 4 | ✅ 相同 |

---

## 🚀 作业提交状态

### Stage 1: 蒸馏训练（3臂，并行）

| ARM | Job ID | 状态 | 节点 |
|-----|--------|------|------|
| base | 17451262 | PENDING | - |
| grad_ascent | 17451263 | PENDING | - |
| esd | 17451264 | PENDING | - |

**预计启动**: 排队中（Priority）  
**预计时长**: 8-10 小时/臂

### Stage 2: 视频生成（依赖 Stage 1）

| ARM | Job ID | 状态 | 依赖 |
|-----|--------|------|------|
| base_distill | 17451265 | PENDING | afterok:S1 |
| grad_ascent_distill | 17451266 | PENDING | afterok:S1 |
| esd_distill | 17451267 | PENDING | afterok:S1 |

**预计时长**: ~1.5 小时

### Stage 3: 安全评估（依赖 Stage 2）

| Job ID | 状态 | 依赖 |
|--------|------|------|
| 17451268 | PENDING | afterok:S2 |

**预计时长**: ~1 小时

---

## 🔧 问题修复记录

### 问题 1: 账户错误
- **错误**: `berzelius-2024-502` 无效
- **修复**: 改为 `Berzelius-2026-243`
- **影响文件**: exp017_distill.sbatch, exp017_submit_all.sh

### 问题 2: 缺少参数
- **错误**: `unrecognized arguments: --use_gradient_checkpointing`
- **修复**: 在 exp017_distill_erased.py 添加该参数
- **提交版本**: v2 (Job 17451262-17451268)

---

## 📊 监控设置

**自动监控已启动** (Task bbav75zk6):
- 检查日志文件生成
- 检测训练开始（Epoch 关键词）
- 监控错误信息
- 每 60 秒检查一次

**手动监控命令**:
```bash
# 查看所有作业
squeue -u $USER | grep exp017

# 实时监控
watch -n 10 'squeue -u $USER | grep exp017'

# 查看日志
tail -f slurm/logs/exp017_distill_17451262_*.out  # base
tail -f slurm/logs/exp017_distill_17451263_*.out  # grad_ascent
tail -f slurm/logs/exp017_distill_17451264_*.out  # esd
```

---

## 📁 文件清单

**所有文件位置正确** ✅

```
fine_tuing_using_DiffSynth-Studio/
├── scripts/
│   ├── distill_wan5b.py                    # 通用蒸馏
│   └── exp017_distill_erased.py            # Exp017（已修复）
├── slurm/
│   ├── exp017_distill.sbatch               # 蒸馏作业（已修复账户）
│   └── exp017_submit_all.sh                # 提交脚本（已修复账户）
└── memory/
    ├── exp017_plan.md                      # 完整设计
    ├── exp017_quickref.md                  # 快速参考
    ├── exp017_implementation_summary.md    # 实现总结
    └── exp017_execution_status.md          # 本文件
```

---

## 🎯 预期结果

### 关键假设

**H1**: 蒸馏削弱擦除效果
- erased_distill 违规率介于 base 和 erased 之间

**H2**: ESD 更抗蒸馏
- esd_distill 安全保持率 > grad_ascent_distill

**H3**: 速度显著提升
- 4步 vs 30步 ≈ 7倍加速

### 预期指标

| 臂 | 步数 | violation_rate | 安全保持率 | 推理时间 |
|----|------|----------------|-----------|----------|
| base | 30 | 0.40 | 0% | 140s |
| base_distill | 4 | 0.40 | 0% | 20s |
| grad_ascent_erased | 30 | 0.15 | 100% | 140s |
| grad_ascent_distill | 4 | 0.25? | 60%? | 20s |
| esd_erased | 30 | 0.10 | 100% | 140s |
| esd_distill | 4 | 0.18? | 73%? | 20s |

---

## ⏱️ 时间线

- **15:44** - 首次提交（账户错误）
- **15:45** - 作业启动，发现参数错误
- **15:46** - 修复参数，取消旧作业
- **15:47** - 重新提交 v2
- **15:48** - 作业排队中
- **预计 16:00** - 作业开始运行
- **预计次日 02:00** - Stage 1 完成（8-10h）
- **预计次日 03:30** - Stage 2 完成（+1.5h）
- **预计次日 04:30** - Stage 3 完成（+1h）

**总预计时长**: ~11-13 小时

---

## 📖 文档位置

- 完整设计: `memory/exp017_plan.md`
- 快速参考: `memory/exp017_quickref.md`
- 实现总结: `memory/exp017_implementation_summary.md`
- **执行状态**: `memory/exp017_execution_status.md` (本文件)

---

## ✅ 下一步

1. **等待作业启动** (~15分钟)
2. **验证训练正常** (监控器自动通知)
3. **等待 Stage 1 完成** (~10小时)
4. **等待 Stage 2-3 自动执行** (~2.5小时)
5. **查看评估结果**: `outputs/exp017_evaluation/*.json`
6. **记录到 experiments.md**

---

**状态更新**: 作业已正常提交，监控器运行中，等待节点分配...
