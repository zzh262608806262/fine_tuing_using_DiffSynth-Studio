# Exp019 Phase 3 进行中

**更新时间**: 2026-09-16 05:00  
**当前状态**: LoRA 合并作业运行中

---

## 当前进度

### ✅ Phase 2 完成 - 擦除训练
- Job 17530967: 成功完成（12分钟）
- 输出: 7个checkpoints，159M
- adapter_step600.pt: 23M（用于合并）

### 🔄 Phase 3 进行中 - LoRA 合并
- Job 17530992 (base): RUNNING
- Job 17530993 (erased): RUNNING
- 预计时间: ~5-10分钟

### ⏳ 待修复 - 其他脚本
所有后续脚本需要修复：
1. ✅ 合并脚本 - 已修复
2. ❌ 生成脚本 - 待修复  
3. ❌ 评估脚本 - 待修复
4. ✅ 微调脚本 - 已修复

---

## 已发现的问题

### 1. 环境模块错误
- ❌ `module load Mambaforge/24.1.2-0` - 不存在
- ✅ 使用 VU `.venv` 环境

### 2. 合并脚本参数错误
- ❌ `--base_model` / `--lora_checkpoint`
- ✅ `--adapter` / `--base-dir` / `--output-dir` / `--bridge-output-dir`

### 3. 微调脚本参数错误
- ❌ 使用错误的参数格式
- ✅ 使用 `accelerate launch` + 正确参数

---

## 下一步

### 合并完成后（~5分钟）：
1. 验证模型生成
   ```bash
   ls -lh models/wan5b/exp019_graddiff_base/
   ls -lh models/wan5b/exp019_graddiff_erased/
   ```

2. 修复并提交其他脚本
   - 生成脚本（baseline + finetune后）
   - 评估脚本（NudeNet）
   - 微调训练（24小时）

---

## 配置总结

所有脚本需要：
- ✅ 使用 VU `.venv` 环境
- ✅ 设置 `set -euo pipefail`
- ✅ 使用正确的参数名称
- ✅ 移除不存在的module load命令

---

**备注**: Phase 2 成功，Phase 3 进行中，剩余脚本需要系统修复。
