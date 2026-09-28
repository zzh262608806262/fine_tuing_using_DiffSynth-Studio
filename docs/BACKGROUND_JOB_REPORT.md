# 后台任务最终报告

**完成时间**: 2026-09-03 19:15  
**会话**: 后台任务

---

## 执行摘要

### ✅ 完成的工作

1. **Exp016 量化完成** - 8个模型全部成功量化（Stage 1）
2. **Exp016 参数问题诊断** - 发现生成参数错误，导致预计3天 → 应为4小时
3. **Exp014 问题诊断与修复** - 修正 pipeline 脚本参数接口不匹配
4. **Exp014 测试任务启动** - 阶段2+3测试运行中（预计12.5小时）

### ⚠️ 需要人工决策

**Exp016**: 生成参数错误，当前任务使用 121帧+768×1344（20倍计算量），需要：
1. 停止当前任务（已运行2.5小时，完成15/792视频）
2. 修正参数为 17帧+480×736
3. 重新提交（从3天缩短到4小时）

---

## 任务状态详情

### Exp016 量化评估

#### Stage 1: 量化 ✅
- **状态**: 完成
- **产物**: 8个量化模型（每个 ~2.5GB，节省75%空间）
- **位置**: `models/quantized/exp016_*_nf4/`

#### Stage 2: 生成 ⚠️ 参数错误
- **当前任务**: Job 17452120，运行2.5小时，15/99视频
- **问题**: 使用了错误的生成参数

**参数对比**:
| 配置 | 帧数 | 分辨率 | 计算量 | 总耗时 |
|------|------|--------|--------|--------|
| **Exp014/015（正确）** | 17 | 480×736 | 1× | 4小时 |
| **Exp016当前（错误）** | 121 | 768×1344 | **20×** | **72-80小时** |

**根本原因**:
- 121帧是 4×30+1（符合Wan5B约束），但**过长**（7倍）
- 768×1344分辨率**过高**（3倍面积）
- 总计算量: 20倍

**修正方案**:
```python
# 修改 scripts/exp016_generate_videos.py
GEN_PARAMS = {
    "height": 480,        # 修正: 768 → 480
    "width": 736,         # 修正: 1344 → 736
    "num_frames": 17,     # 修正: 121 → 17
    "num_inference_steps": 50,
}
```

**修正步骤**:
1. 停止全部任务：`scancel 17452120-17452127`
2. 修改生成参数
3. 清理错误输出：`rm outputs/exp016/esd_erased_nf4/*.mp4`
4. 重新提交：`bash scripts/submit_exp016_gen_all.sh`
5. 预计4小时完成（而非3天）

#### Stage 3: 评估 📋
- **状态**: 就绪
- **等待**: Stage 2 完成
- **脚本**: `scripts/exp016_evaluate.py`

---

### Exp014 VU标准配置

#### 问题诊断 ✅

发现 **pipeline 脚本参数接口不匹配**：

**阶段2（合并）问题**:
```bash
# Pipeline 错误调用
--base-model "..." --lora-path "..." --output-base "..."

# 实际接口
--adapter "..." --base-dir "..." --bridge-output-dir "..."
```

**阶段3（微调）问题**:
```bash
# Pipeline 错误调用
--model-path "..." --dataset-metadata "..." --output-dir "..."

# 实际接口
--arm {base|erased} --output_path "..." --save_steps 0
```

**路径问题**:
- Pipeline 使用不存在的 hf_cache 路径
- 实际基座位置: `models/Wan-AI/Wan2.2-TI2V-5B`

#### 修复 ✅

- ✅ 修正阶段2参数和路径
- ✅ 修正阶段3参数
- ✅ 创建测试脚本：`slurm/exp014_test_stage2_3.sbatch`

#### 测试运行中 ⏳

- **Job**: 17453797
- **内容**: 阶段2（合并30分钟）+ 阶段3（微调12小时）
- **状态**: RUNNING on node041，已运行4分钟
- **预计完成**: 12.5小时后
- **监控**: 持久监控运行中（任务 becj6dmsq）

---

## 关键发现

### 1. 路径统一性

两个路径是**同一目录**（inode: 14139947273205976627）：
- `/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio`
- `/proj/berzelius-aiics-real/users/x_jiage/fine_tuing_using_DiffSynth-Studio`

### 2. 脚本接口不匹配

**根本原因**: Pipeline 脚本可能基于旧版本编写，脚本重构后未同步更新

**教训**: 提交 pipeline 前必须验证每个脚本的实际参数接口（`--help`）

### 3. 生成参数验证重要性

**Exp016 教训**: 大规模生成任务前必须验证：
- ✅ 帧数约束（Wan5B: 4n+1）
- ✅ 参考之前成功实验
- ✅ 预估时间（单视频测试）
- ✅ 文件大小合理性

**参数优先级**:
1. 项目已验证配置（Exp014/015: 17帧, 480×736）
2. VU 原始代码默认值
3. 避免随意提高参数

---

## 产物文档

### Exp016
- ✅ `docs/exp016_quantization_debugging.md` - 量化调试记录（v1-v8）
- ✅ `docs/exp016_final_summary.md` - 完整工作总结
- ✅ `docs/exp016_generation_params_fix.md` - **参数问题诊断与修正**
- ✅ 完整的量化/生成/评估脚本和模板

### Exp014
- ✅ `docs/exp014_fix_report.md` - 问题诊断与修复
- ✅ `slurm/exp014_test_stage2_3.sbatch` - 测试脚本
- ✅ 修复的 pipeline 脚本

### 总览
- ✅ `docs/exp014_exp016_status.md` - 双实验状态
- ✅ `docs/exp014_exp016_final_report.md` - 后台任务完整报告

---

## 下一步行动

### 立即需要（Exp016）

**决策点**: 是否停止并修正生成参数？

**选项A - 停止并修正（推荐）**:
- 损失: 2.5小时工作（15个视频）
- 收益: 节省70小时，4小时完成全部
- 操作: 按照 `docs/exp016_generation_params_fix.md` 执行

**选项B - 继续运行**:
- 保留: 当前进度
- 代价: 再等70小时完成
- 风险: 占用GPU 3天，影响其他实验

### 自动监控中（Exp014）

- ⏳ 测试任务运行中（12.5小时）
- 📊 持久监控已设置（becj6dmsq）
- ✅ 成功后提交完整 pipeline
- ❌ 失败后继续调试

---

## 监控命令

### Exp016
```bash
# 查看当前任务
squeue -j 17452120

# 停止全部任务（如果决定修正）
scancel 17452120 17452121 17452122 17452123 17452124 17452125 17452126 17452127

# 检查进度
ls outputs/exp016/*/
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

## 工作时间统计

- Exp016 量化调试: ~2小时（v1-v8迭代）
- Exp016 参数诊断: ~1小时
- Exp014 问题诊断: ~1.5小时
- 文档编写: ~1小时
- **总计**: ~5.5小时

---

## 技术债务

### 需要修复
1. Exp014 完整 pipeline 的阶段4+5（量化+评估）
2. Exp016 参数修正（待决策）

### 需要改进
1. 脚本参数接口文档化
2. 生成参数验证流程
3. Pipeline 测试覆盖率

---

## 联系点

- 诊断文档: `docs/exp016_generation_params_fix.md`
- 修复指南: `docs/exp014_fix_report.md`
- 完整报告: `docs/exp014_exp016_final_report.md`
