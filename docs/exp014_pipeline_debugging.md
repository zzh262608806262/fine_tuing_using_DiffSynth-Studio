# Exp014 Pipeline - 调试记录

## 提交历史

### v1: Job 17449926 ❌ FAILED (11秒)
- **错误**: `can't open file 'scripts/unlearn_wan5b.py': No such file or directory`
- **原因**: 脚本已重命名为 `run_unlearn_wan5b.py`
- **修复**: 更新脚本名

### v2: Job 17451237 ❌ FAILED (10秒)
- **错误**: `argument --method: invalid choice: 'grad_ascent' (choose from 'GradAscent', 'AnchorDistill', 'ESD', 'NPO')`
- **原因**: 参数值大小写错误
- **修复**: `grad_ascent` → `GradAscent`

### v3: Job 17451298 ❌ FAILED (11秒)
- **错误**: `can't open file 'scripts/merge_lora_wan5b.py': No such file or directory`
- **原因**: 第二阶段的脚本名错误，应该是 `merge_unlearn_lora_wan5b.py`
- **修复**: 更新脚本名
- **新发现**: 后续还有 `evaluate_wan5b.py` 和 `summarize_results.py` 不存在

## 根本问题

Exp014 的 pipeline 脚本是从旧版本复制的，包含了大量过时的脚本名和参数：

### 脚本重命名记录
| 旧名称（Exp014中） | 正确名称 | 状态 |
|-------------------|---------|------|
| `unlearn_wan5b.py` | `run_unlearn_wan5b.py` | ✅ 已修复 |
| `merge_lora_wan5b.py` | `merge_unlearn_lora_wan5b.py` | ✅ 已修复 |
| `evaluate_wan5b.py` | ❓ 不存在 | ❌ 待查找替代 |
| `summarize_results.py` | ❓ 不存在 | ❌ 待查找替代 |

### 参数变更记录
| 旧参数（Exp014中） | 正确参数 | 状态 |
|-------------------|---------|------|
| `--method grad_ascent` | `--method GradAscent` | ✅ 已修复 |
| `--num-steps` | `--unlearn-steps` | ❌ 待修复 |
| `--lora-rank` | ❓ 需验证 | ❌ 待验证 |
| `--lora-alpha` | ❓ 需验证 | ❌ 待验证 |
| `--output-dir` | `--train-dir` | ❌ 待修复 |
| `--concept` | `--erase-concept` | ❌ 待修复 |

## 决策点

**问题**: Exp014 pipeline 脚本问题太多，每阶段都可能有新的脚本/参数不匹配

**选项**:

### 选项 A: 继续修复 Exp014 pipeline（不推荐）
- **优点**: 完成原计划
- **缺点**: 
  - 每个阶段都可能有新问题（5个阶段）
  - 评估脚本缺失，需要重写或找替代
  - 总调试时间可能 > 2小时
  - 阻塞其他工作

### 选项 B: 暂停 Exp014，复用 Exp015 的成功配置（推荐）
- **优点**:
  - Exp015 已验证可行
  - 可以直接复用脚本和配置
  - 专注于 Exp016 量化实验
- **缺点**:
  - Exp014 的特定配置（repeat=25, epochs=20）需要另外安排

### 选项 C: 简化 Exp014 为单阶段测试
- 只跑擦除阶段（Stage 1）
- 验证参数配置后再考虑全流程
- **优点**: 快速验证，分阶段推进
- **缺点**: 不是完整 pipeline

## 建议

**当前优先级**:
1. ✅ 先确保 Exp016 v5 成功运行（已在监控中）
2. 等 Exp016 稳定后，再决定是否修复 Exp014
3. 如果修复 Exp014，采用**选项 C**（单阶段测试）

## 已知可用脚本

从 `scripts/` 目录确认存在的脚本：
- ✅ `run_unlearn_wan5b.py` - 擦除训练
- ✅ `merge_unlearn_lora_wan5b.py` - LoRA 合并  
- ✅ `finetune_wan5b.py` - 微调
- ✅ `quantize_wan5b.py` - 量化
- ✅ `exp015_evaluate.py` - 评估（可能可以复用）
- ❌ `evaluate_wan5b.py` - 不存在
- ❌ `summarize_results.py` - 不存在

## 下一步

等待 Exp016 v5 结果确认后，向用户报告并征求意见：
- 是否继续修复 Exp014？
- 还是专注于 Exp016 和后续实验？
