# Exp014 修复与启动报告

**更新时间**: 2026-09-03 18:37  
**状态**: 测试运行中 ⏳

---

## 问题诊断与修复

### 根本原因

Exp014 pipeline 脚本使用了**错误的脚本参数接口**，导致连续失败：

1. **合并脚本参数不匹配**:
   - Pipeline 调用: `--base-model`, `--lora-path`, `--output-base`, `--output-erased`
   - 实际接口: `--adapter`, `--base-dir`, `--output-dir`, `--bridge-output-dir`

2. **微调脚本参数不匹配**:
   - Pipeline 调用: `--model-path`, `--dataset-metadata`, `--output-dir`, `--save-every-epoch`
   - 实际接口: `--arm {base,erased}`, `--dataset_metadata_path`, `--output_path`, `--save_steps`

3. **路径不匹配**:
   - Pipeline 输出: `models/wan5b/exp014_{base,erased}`
   - 微调脚本期望: `models/Wan-AI/Wan2.2-TI2V-5B{,-erased}`

### 修复方案

**阶段2（合并）修复**:
```bash
# 正确的合并调用
python scripts/merge_unlearn_lora_wan5b.py \
    --adapter "models/unlearn/exp014_wan5b_nudity_600steps/adapter_final.pt" \
    --base-dir "<hf_cache>/transformer" \
    --output-dir "models/unlearn/exp014_wan5b_nudity_600steps/merged" \
    --bridge-output-dir "models/Wan-AI/Wan2.2-TI2V-5B-erased"

# base 模型转换（如果不存在）
python scripts/convert_wan_5b_bridge.py \
    --input-dir "<hf_cache>/transformer" \
    --output-dir "models/Wan-AI/Wan2.2-TI2V-5B"
```

**阶段3（微调）修复**:
```bash
# 正确的微调调用
python scripts/finetune_wan5b.py \
    --arm base \
    --dataset_repeat 25 \
    --num_epochs 20 \
    --lora_rank 32 \
    --learning_rate 1e-4 \
    --output_path "models/finetune/exp014_base_ft" \
    --save_steps 0

python scripts/finetune_wan5b.py \
    --arm erased \
    --dataset_repeat 25 \
    --num_epochs 20 \
    --lora_rank 32 \
    --learning_rate 1e-4 \
    --output_path "models/finetune/exp014_erased_ft" \
    --save_steps 0
```

---

## 当前测试

### 测试任务
- **Job ID**: 17453789
- **脚本**: `slurm/exp014_test_stage2_3.sbatch`
- **内容**: 阶段2（合并）+ 阶段3（微调）
- **时间限制**: 16小时
- **状态**: ⏳ RUNNING（node042）
- **监控**: 持久监控已启动（16小时超时）

### 预计耗时
- **阶段2（合并）**: ~30分钟
  - 合并擦除LoRA: ~20分钟
  - 转换base模型: ~10分钟（如果需要）
- **阶段3（微调）**: ~12小时
  - base微调: ~6小时
  - erased微调: ~6小时

**总计**: ~12.5小时

---

## 路径说明

### 两个项目路径是同一个目录

通过 inode 检查确认：
- `/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio`
- `/proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio`

**inode**: `14139947273205976627`（相同）

这两个路径指向**同一个目录**，可能 `/home/x_jiage/jiage/` 是 `/proj/.../x_jiage/` 的挂载点。

### 文件结构

```
models/
├── unlearn/
│   └── exp014_wan5b_nudity_600steps/    ✅ 已完成（阶段1）
│       ├── adapter_final.pt
│       ├── adapter_step{100,200,...,600}.pt
│       └── merged/                       ⏳ 生成中（阶段2）
├── Wan-AI/
│   ├── Wan2.2-TI2V-5B/                   ⏳ 生成中（阶段2，base）
│   └── Wan2.2-TI2V-5B-erased/            ⏳ 生成中（阶段2，erased）
└── finetune/
    ├── exp014_base_ft/                   ⏳ 待生成（阶段3）
    └── exp014_erased_ft/                 ⏳ 待生成（阶段3）
```

---

## Exp014 完整实验设计

### 研究目标
采用 VU 标准配置（rank=8, 600步）+ 细粒度微调（repeat=25 × epochs=20），分析擦除能力在微调过程中的"回潮曲线"。

### 实验配置
- **擦除**: GradAscent, rank=8, 600步
- **微调**: repeat=25 × epochs=20 = 500次曝光/样本
- **checkpoint**: 20个微调点（每个epoch保存）
- **量化**: NF4（42个模型）
- **评估**: NudeNet（threshold=0.6）

### 四臂设计
1. `base`: 未擦除基座
2. `erased`: nudity擦除（600步）
3. `base_ft`: base + 微调（20个checkpoint）
4. `erased_ft`: erased + 微调（20个checkpoint）

### 预期产物
- 擦除LoRA: 7个checkpoint（100/200/.../600 + final）
- 合并模型: base, erased
- 微调LoRA: 2×20个checkpoint
- 量化模型: 42个（base + erased + 20×base_ft + 20×erased_ft）
- 评估结果: 42组安全性指标

---

## 下一步

### 测试完成后
1. ✅ 如果测试成功 → 提交完整 pipeline（包括量化+评估）
2. ❌ 如果测试失败 → 检查日志，继续修复

### 待修复的阶段
- **阶段4（量化）**: 需要确认 `quantize_wan5b.py` 的参数接口
- **阶段5（评估）**: 需要确认 `evaluate_wan5b.py` 的参数接口

---

## 监控命令

```bash
# 查看测试任务状态
squeue -j 17453789

# 查看实时日志
tail -f logs/exp014_stage2_3_17453789.out

# 检查合并进度
ls -lh models/Wan-AI/Wan2.2-TI2V-5B-erased/

# 检查微调进度
ls -lh models/finetune/exp014_base_ft/ models/finetune/exp014_erased_ft/
```

---

## 文档更新

- ✅ 修复的 pipeline: `slurm/exp014_wan5b_pipeline.sbatch`（阶段2+3）
- ✅ 测试脚本: `slurm/exp014_test_stage2_3.sbatch`
- 📝 本报告: `docs/exp014_fix_report.md`
