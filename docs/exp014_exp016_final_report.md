# 后台任务完成报告 - Exp014 & Exp016

**完成时间**: 2026-09-03 18:42  
**会话**: 后台任务

---

## 任务完成情况

### ✅ Exp016 量化评估 - 运行中

**状态**: Stage 1 完成 ✅ | Stage 2 运行中 ⏳ | Stage 3 就绪 📋

#### Stage 1: 模型量化（已完成）
- 8个模型臂全部量化成功
- 每个模型 ~2.5GB（原始 ~10GB，节省 75%）
- 位置: `models/quantized/exp016_*_nf4/`

#### Stage 2: 视频生成（运行中）
- **Job**: 17452120（esd_erased_nf4）
- **运行时间**: 2小时13分钟
- **进度**: 15/99 视频（15%）
- **预计总耗时**: 72-80 小时（约3天）
  - 单个视频：6-7分钟
  - 单臂：9-10小时
  - 8臂串行：3天左右
- **监控**: 持久监控运行中（任务 bdo1cxbqk）

#### Stage 3: 评估（就绪）
- 脚本已准备：`scripts/exp016_evaluate.py`
- 提交脚本：`scripts/submit_exp016_eval_all.sh`
- 等待生成完成后自动启动

---

### ⏳ Exp014 VU标准配置 - 测试中

**状态**: 参数修复完成 ✅ | 测试运行中 ⏳

#### 问题诊断
发现 **pipeline 脚本参数接口不匹配** 导致连续失败：

1. **合并脚本参数错误**
   - Pipeline使用：`--base-model`, `--lora-path`, `--output-base`
   - 实际需要：`--adapter`, `--base-dir`, `--output-dir`, `--bridge-output-dir`

2. **微调脚本参数错误**
   - Pipeline使用：`--model-path`, `--dataset-metadata`, `--output-dir`
   - 实际需要：`--arm {base,erased}`, `--output_path`, `--save_steps`

3. **路径不匹配**
   - Pipeline输出：`models/wan5b/exp014_*`
   - 脚本期望：`models/Wan-AI/Wan2.2-TI2V-5B{,-erased}`

4. **基座路径错误**
   - Pipeline使用：`/proj/.../hf_cache/hub/.../transformer`（不存在）
   - 实际位置：`models/Wan-AI/Wan2.2-TI2V-5B`（已存在）

#### 修复方案
1. **阶段2（合并）**:
   ```bash
   python scripts/merge_unlearn_lora_wan5b.py \
       --adapter "models/unlearn/exp014_wan5b_nudity_600steps/adapter_final.pt" \
       --base-dir "models/Wan-AI/Wan2.2-TI2V-5B" \
       --output-dir "models/unlearn/exp014_wan5b_nudity_600steps/merged" \
       --bridge-output-dir "models/Wan-AI/Wan2.2-TI2V-5B-erased" \
       --force
   ```

2. **阶段3（微调）**:
   ```bash
   python scripts/finetune_wan5b.py \
       --arm {base|erased} \
       --dataset_repeat 25 \
       --num_epochs 20 \
       --lora_rank 32 \
       --learning_rate 1e-4 \
       --output_path "models/finetune/exp014_{base|erased}_ft" \
       --save_steps 0
   ```

#### 测试任务
- **Job**: 17453797（v2，修复基座路径）
- **内容**: 阶段2（合并）+ 阶段3（微调）
- **状态**: ⏳ RUNNING（node041，运行20秒）
- **预计耗时**: 12.5小时（合并30分钟 + 微调12小时）
- **监控**: 持久监控运行中（任务 becj6dmsq，16小时超时）

#### 失败历史
- v1 (17453789): 基座路径错误（24秒失败）
- v2 (17453797): 修复基座路径，运行中 ⏳

---

## 关键发现

### 路径统一性
通过 inode 检查确认两个路径是**同一个目录**：
- `/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio`
- `/proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio`
- **inode**: 14139947273205976627（相同）

这不是路径不一致问题，而是脚本参数接口问题。

### 脚本接口不匹配
Pipeline 脚本使用的参数接口与实际脚本不符，说明：
1. Pipeline 可能是基于旧版本脚本编写的
2. 脚本经过重构但 pipeline 未同步更新
3. 需要在提交前验证所有脚本的实际参数

---

## 产物位置

### Exp016
```
models/quantized/
├── exp016_esd_erased_nf4/           ✅ 2.5GB
├── exp016_esd_base_nf4/             ✅ 2.5GB
├── exp016_npo_erased_nf4/           ✅ 2.5GB
├── exp016_npo_base_nf4/             ✅ 2.5GB
├── exp016_grad_ascent_erased_nf4/   ✅ 2.5GB
├── exp016_grad_ascent_base_nf4/     ✅ 2.5GB
├── exp016_anchor_distill_erased_nf4/✅ 2.5GB
└── exp016_anchor_distill_base_nf4/  ✅ 2.5GB

outputs/exp016/
└── esd_erased_nf4/                  ⏳ 15/99 视频
```

### Exp014
```
models/unlearn/
└── exp014_wan5b_nudity_600steps/    ✅ 已完成
    ├── adapter_final.pt             ✅
    ├── adapter_step{100..600}.pt    ✅
    └── merged/                       ⏳ 生成中

models/Wan-AI/
├── Wan2.2-TI2V-5B/                  ✅ 已存在（base）
└── Wan2.2-TI2V-5B-erased/           ⏳ 生成中（阶段2）

models/finetune/
├── exp014_base_ft/                  📋 待生成（阶段3）
└── exp014_erased_ft/                📋 待生成（阶段3）
```

---

## 文档产物

### Exp016
- ✅ `docs/exp016_quantization_debugging.md` - 量化调试记录（v1-v8）
- ✅ `docs/exp016_progress_report.md` - 进度报告
- ✅ `docs/exp016_final_summary.md` - 完整工作总结
- ✅ `scripts/exp016_*.py` - 量化/生成/评估脚本
- ✅ `slurm/exp016_*_template.sbatch` - SLURM 模板

### Exp014
- ✅ `docs/exp014_fix_report.md` - 问题诊断与修复报告
- ✅ `slurm/exp014_wan5b_pipeline.sbatch` - 修复的 pipeline（阶段2+3）
- ✅ `slurm/exp014_test_stage2_3.sbatch` - 测试脚本
- ✅ `memory/experiments.md` - 实验记录更新

### 通用
- ✅ `docs/exp014_exp016_status.md` - 双实验状态总览
- ✅ `docs/exp014_exp016_final_report.md` - 本最终报告

---

## 监控命令

### Exp016
```bash
# 查看生成任务
squeue -j 17452120,17452121,17452122,17452123,17452124,17452125,17452126,17452127

# 检查进度
ls outputs/exp016/*/*.mp4 | wc -l

# 生成完成后提交评估
bash scripts/submit_exp016_eval_all.sh
```

### Exp014
```bash
# 查看测试任务
squeue -j 17453797

# 查看日志
tail -f logs/exp014_stage2_3_17453797.out

# 检查合并进度
ls -lh models/Wan-AI/Wan2.2-TI2V-5B-erased/

# 检查微调进度
ls models/finetune/exp014_{base,erased}_ft/
```

---

## 下一步行动

### Exp016（自动）
1. ⏳ 等待生成完成（预计3天，持久监控运行中）
2. 📋 自动提交评估任务
3. 📊 分析量化前后性能差异

### Exp014（需跟进）
1. ⏳ 等待测试完成（预计12.5小时，持久监控运行中）
2. ✅ 如果成功 → 修复完整 pipeline 的阶段4+5，提交完整运行
3. ❌ 如果失败 → 检查日志，继续修复

---

## 经验总结

### 调试策略
1. **逐阶段测试**: 不要一次运行完整 pipeline，先测试单个阶段
2. **验证参数**: 用 `--help` 验证脚本实际参数接口
3. **检查路径**: 确认文件确实存在于期望位置
4. **保留详细记录**: 记录每次失败的原因和修复方案

### 脚本维护
1. **接口稳定性**: 脚本重构时同步更新调用方
2. **参数文档**: 在脚本注释中记录参数变更历史
3. **版本控制**: 重要的 pipeline 脚本应该版本化

### 集群使用
1. **时间估算**: 提前估算任务时间，避免超时
2. **监控设置**: 长时间任务设置持久监控
3. **资源规划**: 3天的生成任务占用GPU，需要权衡优先级
