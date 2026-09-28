# 实验数据快速查询手册

**最后更新**: 2026-09-16  
**用途**: AI 对话开始时快速了解已有数据

---

## 📊 数据完整性清单

| 实验 | 视频数 | 评估完成 | 位置 |
|------|--------|----------|------|
| Exp015 | 792 (8臂×99) | ✅ | `outputs/exp015/`, `outputs/exp015_evaluation/` |
| Exp016 | 792 (8臂×99) | ✅ | `outputs/exp016/`, `outputs/exp016_evaluation/` |
| Exp017 | 495 (5臂×99) | ❌ | `outputs/exp017/` (缺评估) |
| Exp018 | 396 (4臂×99) | ✅ | `outputs/exp018/`, `outputs/exp018/evaluation/` |

**总计**: 2,475 个视频，2,079 个已评估

---

## 🔍 Exp015 - 基线数据（擦除效果）

**位置**: `outputs/exp015_evaluation/*.json`

### 违规率汇总 (NudeNet threshold=0.6)

| 方法 | Base | Erased | 效果 |
|------|------|--------|------|
| ESD | 39.4% | 34.3% | ✅ -5.1pp |
| AnchorDistill | 40.4% | 38.4% | ✅ -2.0pp |
| GradAscent | 37.4% | 42.4% | ❌ +5.1pp |
| NPO | 35.4% | 41.4% | ❌ +6.1pp |

**结论**: ESD 擦除效果最好，GradAscent/NPO 失效

---

## 🔍 Exp016 - 量化影响

**位置**: `outputs/exp016_evaluation/*.json`

### 量化后违规率对比

| 方法 | 原始Erased | 量化后 | 变化 |
|------|-----------|--------|------|
| ESD | 34.3% | 26.3% | ✅ -8.1pp |
| NPO | 41.4% | 36.4% | ✅ -5.1pp |
| AnchorDistill | 38.4% | 40.4% | +2.0pp |
| GradAscent | 42.4% | 43.4% | +1.0pp |

**结论**: 量化意外改善了 ESD/NPO 的安全性

---

## 🔍 Exp018 - 微调回潮

**位置**: `outputs/exp018/evaluation/*.json`

### 微调后回潮幅度

| 方法 | Erased | 微调e19 | 回潮 |
|------|--------|---------|------|
| AnchorDistill | 38.4% | 46.5% | +8.1pp |
| GradAscent | 42.4% | 48.5% | +6.1pp |
| NPO | 41.4% | 48.5% | +7.1pp |
| **ESD** | 34.3% | 44.4% | **+10.1pp** |

**结论**: 所有方法都出现显著回潮（6-10pp），ESD 回潮最严重

---

## 🔍 Exp017 - 蒸馏效果（待评估）

**位置**: `outputs/exp017/`

### 现状
- ✅ 5臂视频已生成（495个）
- ❌ 缺少 NudeNet 评估结果
- 📋 需要操作: 运行评估脚本

### 臂列表
1. `base_distill` - 原始模型蒸馏（99个视频）
2. `grad_ascent_distill` - GradAscent擦除后蒸馏（99个视频）
3. `esd_distill` - ESD擦除后蒸馏（99个视频）
4. `npo_distill` - NPO擦除后蒸馏（99个视频）
5. `anchor_distill_distill` - AnchorDistill擦除后蒸馏（99个视频）

---

## 📈 跨实验对比 - GradAscent 方法

| 状态 | 违规率 | 来源 |
|------|--------|------|
| Base原始 | 37.4% | Exp015 |
| 擦除后 | 42.4% | Exp015 |
| 量化后 | 43.4% | Exp016 |
| 微调e19后 | 48.5% | Exp018 |

**趋势**: GradAscent 擦除失败，微调后进一步恶化

---

## 📈 跨实验对比 - ESD 方法

| 状态 | 违规率 | 来源 |
|------|--------|------|
| Base原始 | 39.4% | Exp015 |
| 擦除后 | 34.3% | Exp015 ✅ 最佳 |
| 量化后 | 26.3% | Exp016 ✅ 更好 |
| 微调e19后 | 44.4% | Exp018 ❌ 回潮 |

**趋势**: ESD 擦除有效，量化提升，但微调后严重回潮

---

## 🔧 快速查询命令

### 检查文件存在
```bash
# 检查所有视频
find outputs/exp01{5,6,7,8} -name "*.mp4" | wc -l
# 应输出: 2475

# 检查评估结果
ls outputs/exp015_evaluation/*.json | wc -l  # 应输出: 8
ls outputs/exp016_evaluation/*.json | wc -l  # 应输出: 8
ls outputs/exp018/evaluation/*.json | wc -l  # 应输出: 4
```

### 读取违规率
```python
import json

# Exp015 - GradAscent 擦除后
data = json.load(open('outputs/exp015_evaluation/grad_ascent_erased_evaluation.json'))
print(f"违规率: {data['nudenet']['summary']['violation_rate']:.1%}")
print(f"违规视频: {data['nudenet']['summary']['violations']}/99")

# Exp018 - GradAscent 微调后
data = json.load(open('outputs/exp018/evaluation/grad_ascent_ft_e19_results.json'))
print(f"违规率: {data['nudenet']['summary']['violation_rate']:.1%}")
```

---

## ⚠️ 重要注意事项

1. **视频长度**: 所有视频只有 **17帧**（约1秒），可能不足以展现完整 unsafe 内容
2. **Exp017 缺评估**: 495 个视频已生成但未评估
3. **擦除配置疑问**: 3/4 方法擦除后违规率上升，可能配置有误
4. **数据可靠性**: 基于现有数据的结论是可靠的，但建议用更长视频验证

---

## 📞 相关文档

- 完整实验记录: `memory/experiments.md`
- 项目总览: `memory/MASTER_SUMMARY.md`
- 详细数据表: `memory/exp015_exp018_baseline_data.md`
