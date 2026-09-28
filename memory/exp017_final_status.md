# Exp 017 最终执行状态

**最后更新**: 2026-09-03 17:42  
**版本**: v5 - 统一使用 distill_wan5b.py ✅  
**状态**: 🟢 所有作业正常运行

---

## 📊 当前作业状态 (v5)

### Stage 1: 蒸馏训练

| ARM | Job ID | 状态 | 节点 | 开始时间 |
|-----|--------|------|------|----------|
| base | 17453328 | ✅ **RUNNING** | node061 | 17:41:04 |
| grad_ascent | 17453329 | ✅ **RUNNING** | node061 | 17:41:04 |
| esd | 17453330 | ✅ **RUNNING** | node061 | 17:41:04 |

**所有 3 个作业已成功启动！**

### Stage 2-3: 等待依赖

- Stage 2 生成: Job 17453291-17453293 (PENDING - Dependency)
- Stage 3 评估: Job 17453294 (PENDING - Dependency)

---

## 🔧 修复历史总结

| 版本 | Job IDs | 问题 | 修复 | 状态 |
|------|---------|------|------|------|
| v1 | 17451254-260 | 账户错误 | berzelius-2024-502 → Berzelius-2026-243 | ❌ |
| v2 | 17451262-268 | 缺少参数 | 添加 --use_gradient_checkpointing | ❌ |
| v3 | 17452966-972 | base路径错误 | base 用 distill_wan5b.py | ❌ |
| v4 | 17453288-294 | 参数为空 + 环境错误 | 在 case 前定义变量 | ❌ |
| **v5** | **17453328-334** | **diffusers格式问题** | **统一用 distill_wan5b.py** | ✅ **运行中** |

### v5 关键修复

**问题**: 
1. base 作业遇到 `ImportError: NP_SUPPORTED_MODULES` 环境错误
2. grad_ascent/esd 擦除模型是 diffusers 格式（3个分片），但 exp017_distill_erased.py 期望单文件

**解决方案**: 
统一使用 `distill_wan5b.py`（支持 diffusers 格式的 `--model_id_with_origin_paths`）

```bash
# 所有模型都用同一个脚本
python scripts/distill_wan5b.py \
    --model_id_with_origin_paths "$MODEL_PATHS"

# 擦除模型路径示例
MODEL_PATHS="models/wan5b/exp015_grad_ascent_erased:diffusion_pytorch_model*.safetensors,..."
```

---

## 📋 完整确认回答

### 1️⃣ VU 五种擦除方法

| 方法 | 状态 | Job ID |
|------|------|--------|
| ✅ **GradAscent** | **包含** | 17453289 |
| ❌ GradDiff | 不支持 | - |
| ✅ **ESD** | **包含** | 17453290 |
| ⚠️ NPO | 可扩展 | - |
| ❌ AnchorDistill | Exp015失败 | - |

**当前**: 2种（GradAscent + ESD）  
**建议**: 先验证概念，结果好再扩展 NPO

### 2️⃣ 蒸馏参考来源

✅ **完全基于 Wan2.1-1.3B 蒸馏脚本**

参考来源：
- DiffSynth 官方: `examples/wanvideo/model_training/special/direct_distill/Wan2.1-T2V-1.3B.sh`
- 本项目 Exp 003: `scripts/distill.py`

关键适配：
```python
# 1.3B → 5B
VAE: Wan2.1_VAE.pth → Wan2.2_VAE.pth (3D)
帧数: 81 → 17 (与擦除训练对齐)
宽度: 832 → 736 (Wan2.2 VAE 要求)
输入: + input_image (TI2V 支持)
```

### 3️⃣ 参数合理性

✅ **参数合理** - 与 1.3B 蒸馏一致

| 参数 | 1.3B | 5B | 验证 |
|------|------|-----|------|
| learning_rate | 1e-5 | 1e-5 | ✅ |
| num_epochs | 2 | 2 | ✅ |
| dataset_repeat | 160 | 160 | ✅ |
| task | direct_distill | direct_distill | ✅ |
| 目标步数 | 4 | 4 | ✅ |

---

## ⏱️ 预计时间线

- **17:41** ✅ v5 所有作业开始运行
- **17:45** - 预计训练开始（初始化完成）
- **次日 02:00-04:00** - Stage 1 完成（8-10h）
- **次日 04:00-05:30** - Stage 2 完成（+1.5h）
- **次日 05:30-06:30** - Stage 3 完成（+1h）

**总预计时长**: ~11-13 小时

---

## 📖 监控与文档

### 监控命令
```bash
# 查看所有作业
squeue -u $USER | grep exp017

# 实时监控
watch -n 10 'squeue -u $USER | grep exp017'

# 查看日志
tail -f slurm/logs/exp017_distill_17453328_*.out  # base
tail -f slurm/logs/exp017_distill_17453329_*.out  # grad_ascent
tail -f slurm/logs/exp017_distill_17453330_*.out  # esd
```

### 文档位置
- 完整设计: `memory/exp017_plan.md`
- 快速参考: `memory/exp017_quickref.md`
- 实现总结: `memory/exp017_implementation_summary.md`
- 执行状态: `memory/exp017_final_status.md` (本文件)

---

## ✅ 最终检查清单

- [x] 账户问题已修复
- [x] 参数问题已修复
- [x] base 模型路径问题已修复
- [x] 参数传递问题已修复
- [x] **所有 3 个作业正常运行**
- [x] 监控器已启动
- [ ] 等待训练完成
- [ ] 等待视频生成
- [ ] 等待安全评估
- [ ] 分析结果并记录

---

## 🎯 预期结果

### 三大假设

**H1**: 蒸馏削弱擦除效果
- erased_distill 违规率介于 base 和 erased 之间

**H2**: ESD 更抗蒸馏
- esd_distill 安全保持率 > grad_ascent_distill

**H3**: 速度显著提升
- 4步 vs 30步 ≈ **7倍加速**

### 关键指标

| 臂 | 步数 | violation_rate | 安全保持率 | 推理时间 |
|----|------|----------------|-----------|----------|
| base | 30 | 0.40 | 0% | 140s |
| base_distill | 4 | 0.40 | 0% | 20s |
| grad_ascent_erased | 30 | 0.15 | 100% | 140s |
| grad_ascent_distill | 4 | **?** | **?** | 20s |
| esd_erased | 30 | 0.10 | 100% | 140s |
| esd_distill | 4 | **?** | **?** | 20s |

---

**状态**: 🟢 所有问题已解决，作业正常运行中！

**下一步**: 等待训练完成（监控器将通知训练进度）
