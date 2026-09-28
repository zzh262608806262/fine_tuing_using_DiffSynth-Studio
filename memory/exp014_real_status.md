# Exp014 真实状态总结

**更新时间**: 2026-09-04 14:00  
**发现**: Exp014 的 Stage 1-2 早已完成（8月30-31日），但后续阶段因 pipeline 脚本问题未能继续

---

## ✅ 已完成的阶段

### Stage 1: 擦除训练 ✅ 完成（8月30日）
**产物**: `models/unlearn/exp014_wan5b_nudity_600steps/`
- ✓ `adapter_step100.pt` ~ `adapter_step600.pt` (6个checkpoint)
- ✓ `adapter_final.pt` (最终 LoRA)
- ✓ `training_protocol.json`, `training_trace.jsonl`
- ✓ 每个文件约 23 MB

### Stage 2: LoRA 合并 ✅ 完成（8月31日）
**产物**:
1. `models/unlearn/exp014_wan5b_nudity_600steps/merged/` - Diffusers 格式
   - ✓ 5个 safetensors 分片（最初合并）
   - ✓ 3个 safetensors 分片（9月3日重新合并）
   - ✓ `merge_report.json` (最新: 9月3日 18:40)

2. `models/wan5b/exp014_base/` - Base 模型 (Diffusers 格式)
   - ✓ 5个 safetensors 分片

3. `models/wan5b/exp014_erased/` - Erased 模型 (Diffusers 格式)
   - ✓ 3个 safetensors 分片 (9.4 GB 总计)

### Stage 3: 微调 ❌ 未完成（仅创建了目录）
**产物**: `models/finetune/exp014_base_ft/`
- ⚠️ 只有 `training_args.json` (1.9 KB)
- ❌ 没有实际的 LoRA checkpoint
- **结论**: 微调任务启动但立即失败

### Stage 4+: 后续阶段 ❌ 未开始
- ❌ 量化 (NF4)
- ❌ 生成评测视频
- ❌ NudeNet 评估

---

## 🔍 问题根源

根据 `docs/exp014_pipeline_debugging.md`：

1. **Pipeline 脚本过时**: 使用了旧版本的脚本名和参数
2. **多次失败尝试**:
   - Job 17449926: `unlearn_wan5b.py` 不存在
   - Job 17451237: 方法名大小写错误
   - Job 17451298: `merge_lora_wan5b.py` 不存在

3. **决策**: 暂停修复，专注于 Exp015/016

---

## 📊 当前状态总结

| 阶段 | 状态 | 完成时间 | 产物 |
|------|------|----------|------|
| Stage 1: 擦除训练 | ✅ 完成 | 2026-08-30 | 7个 adapter checkpoint |
| Stage 2: LoRA 合并 | ✅ 完成 | 2026-08-31 | base + erased 模型 |
| Stage 3: 微调 | ❌ 失败 | 2026-08-31 | 仅有空目录 |
| Stage 4: 量化 | ❌ 未开始 | - | - |
| Stage 5: 生成 | ❌ 未开始 | - | - |
| Stage 6: 评估 | ❌ 未开始 | - | - |

---

## 💡 当前状况

**误操作**: 刚才我提交了重复的 Stage 1+2 测试作业（Job 17460630, 17460631）
- ✅ 已取消这些重复作业
- 原因: 没有仔细检查现有产物就重新提交

**真实需求**: 
- ✅ Stage 1-2 已完成，不需要重做
- ❌ 需要修复 Stage 3-6 的 pipeline 问题

---

## 🔄 接下来应该做什么？

### 选项 A: 继续修复 Exp014（基于已有产物）

从 Stage 3（微调）开始：
1. 创建正确的微调脚本（基于 Exp012）
2. 微调两臂: base_ft + erased_ft
3. 量化 42 个模型
4. 生成 + 评估

**优点**: 
- Stage 1-2 已完成，可以复用
- 获得细粒度的回潮曲线（20个checkpoint）

**缺点**:
- 需要修复 pipeline（估计2-3小时）
- 微调 + 量化 + 生成 + 评估可能需要数天

### 选项 B: 暂停 Exp014，专注 Exp016

等待 Exp016 完成（量化评估实验）：
- Exp016 已经在运行中（预计3天完成）
- 先完成 Exp016，再回来处理 Exp014

**建议**: **选项 B**，原因：
1. Exp016 已经在运行，优先确保其完成
2. Exp014 的完整 pipeline 需要大量调试时间
3. 可以在 Exp016 运行期间准备 Exp014 的修复脚本

---

## 📝 下一步行动

1. **立即**: 确认 Exp016 运行正常
2. **短期**: 准备 Exp014 Stage 3-6 的正确脚本（不提交）
3. **中期**: 等 Exp016 完成后，再决定是否继续 Exp014

---

**重要经验**: 
- ✅ 提交前必须先检查已有产物
- ✅ 不要假设阶段未完成就重新运行
- ✅ 仔细阅读状态文档和历史记录
