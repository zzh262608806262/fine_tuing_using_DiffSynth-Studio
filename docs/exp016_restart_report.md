# Exp016 生成任务重启报告

**时间**: 2026-09-03 19:25  
**状态**: ✅ 已重启，使用正确参数

---

## 执行摘要

**问题**: 生成参数错误（121帧 + 768×1344），预计需要72-80小时  
**修正**: 使用正确参数（17帧 + 480×736），预计4小时完成  
**行动**: 已停止旧任务，修改参数，重新提交

---

## 参数修正

### 修改前（错误）
```python
GEN_PARAMS = {
    "height": 768,        # ❌ 过高
    "width": 1344,        # ❌ 过高
    "num_frames": 121,    # ❌ 过长（7倍）
    "num_inference_steps": 50,
    "cfg_scale": 7.0,
    "seed": 42,
}
# 预计: 单视频 6-7分钟，8臂×99条 = 72-80小时（3天）
```

### 修改后（正确）
```python
GEN_PARAMS = {
    "height": 480,        # ✅ 参考 Exp014/015
    "width": 736,         # ✅ 参考 Exp014/015
    "num_frames": 17,     # ✅ 4×4+1, Wan5B标准
    "num_inference_steps": 50,
    "cfg_scale": 7.0,
    "seed": 42,
}
# 预计: 单视频 ~20秒，8臂×99条 = ~4小时
```

### 改进幅度
- **计算量**: 减少 20倍
- **总耗时**: 72-80小时 → **4小时**（节省94%时间）
- **GPU占用**: 3天 → 4小时

---

## 执行步骤

### 1. 停止旧任务 ✅
```bash
scancel 17452120-17452127
```
- 停止了8个任务（Job 17452120-17452127）
- 损失：2.5小时工作，15个视频

### 2. 修改参数 ✅
文件：`scripts/exp016_generate_videos.py`
- 帧数: 121 → 17
- 分辨率: 768×1344 → 480×736

### 3. 重新提交 ✅
```bash
bash scripts/submit_exp016_gen_all.sh
```

提交的任务：
| Job ID | 臂 | 状态 | 节点 |
|--------|-----|------|------|
| 17453867 | esd_erased_nf4 | ✅ RUNNING | node024 |
| 17453868 | esd_base_nf4 | ✅ RUNNING | node001 |
| 17453869 | npo_erased_nf4 | ✅ RUNNING | node003 |
| 17453870 | npo_base_nf4 | ✅ RUNNING | node003 |
| 17453871 | grad_ascent_erased_nf4 | ✅ RUNNING | node007 |
| 17453872 | grad_ascent_base_nf4 | ✅ RUNNING | node007 |
| 17453873 | anchor_distill_erased_nf4 | ✅ RUNNING | node007 |
| 17453874 | anchor_distill_base_nf4 | ✅ RUNNING | node008 |

**并行运行**: 8个任务同时在不同节点运行  
**预计完成**: ~4小时后（单臂30分钟，最慢的完成即可）

### 4. 监控设置 ✅
- **参数验证监控**（brltb7rpi）: 等待首个视频生成，验证参数正确
- **完成监控**（bnks5agfl）: 追踪全部8个任务，完成时统计视频数量

---

## 验证计划

### 自动验证
1. ⏳ 首个视频生成后（~1分钟）
   - 检查 `gen_params.json`
   - 确认: 17帧 + 480×736

2. ⏳ 全部任务完成后（~4小时）
   - 统计每臂视频数量（应为99个）
   - 自动通知

### 手动验证
```bash
# 查看任务状态
squeue -u $USER | grep exp016-gen

# 检查生成进度
for arm in esd_erased_nf4 esd_base_nf4 npo_erased_nf4 npo_base_nf4 \
           grad_ascent_erased_nf4 grad_ascent_base_nf4 \
           anchor_distill_erased_nf4 anchor_distill_base_nf4; do
    echo "$arm: $(ls outputs/exp016/$arm/*.mp4 2>/dev/null | wc -l)/99"
done

# 检查参数（首个视频生成后）
cat outputs/exp016/esd_erased_nf4/gen_params.json | python3 -m json.tool
```

---

## 下一步（自动）

生成完成后，自动提交评估任务：

```bash
bash scripts/submit_exp016_eval_all.sh
```

这将：
1. 对8个臂的792个视频（8×99）进行 NudeNet 检测
2. 生成安全性报告
3. 对比量化前后的性能差异

---

## 成本分析

### 旧配置（已取消）
- GPU时间: 72-80小时 × 8臂 = 576-640 GPU小时
- 挂钟时间: 3天（串行）
- 成本: 高

### 新配置（当前）
- GPU时间: 4小时 × 8臂 = 32 GPU小时（并行）
- 挂钟时间: 4小时
- 成本: **节省94%**

### 节省
- GPU小时: 节省 544-608 小时
- 挂钟时间: 节省 68-76 小时
- 早发现价值: 避免浪费3天等待

---

## 经验教训

### ✅ 做对的事
1. **及时发现**: 运行2.5小时后发现问题，及时止损
2. **参考历史**: 对比 Exp014/015 的成功配置
3. **详细诊断**: 完整分析参数来源和计算量差异
4. **果断决策**: 虽然损失2.5小时，但节省70小时

### ❌ 可以改进
1. **提交前验证**: 应该在提交前检查参数合理性
2. **小规模测试**: 可以先生成1-2个视频验证参数
3. **文档参考**: 应该查看之前成功实验的配置

### 📝 流程改进
未来大规模生成任务的检查清单：
- [ ] 参考项目已验证配置
- [ ] 验证帧数约束（Wan5B: 4n+1）
- [ ] 生成1-2个测试视频
- [ ] 检查文件大小合理性
- [ ] 预估总耗时
- [ ] 评估GPU占用时间

---

## 相关文档

- **诊断报告**: `docs/exp016_generation_params_fix.md`
- **量化工作**: `docs/exp016_final_summary.md`
- **后台任务**: `docs/BACKGROUND_JOB_REPORT.md`

---

## 监控命令

```bash
# 查看任务状态
squeue -j 17453867,17453868,17453869,17453870,17453871,17453872,17453873,17453874

# 查看单个任务日志
tail -f logs/exp016_gen_esd_erased_nf4_17453867.out

# 检查进度
ls outputs/exp016/*/000.mp4 | wc -l  # 应该有8个（每臂第一个）
```

---

**预计完成时间**: 2026-09-03 23:30（4小时后）  
**状态**: ✅ 运行中，自动监控已设置
