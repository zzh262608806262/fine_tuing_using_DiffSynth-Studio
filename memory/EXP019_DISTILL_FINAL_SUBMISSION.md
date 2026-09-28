# Exp019 蒸馏实验 - 最终提交

**提交时间**: 2026-09-21 14:00  
**状态**: ✅ 已提交全量作业

---

## 📊 最终作业链

| 阶段 | Job ID | 状态 | 说明 |
|------|--------|------|------|
| **蒸馏训练** | 17539965 | ✅ 已完成 | 1天22小时，生成10个checkpoint |
| **生成视频** | 17584765 | ⏳ 队列中 | 99个视频，4步推理，1小时 |
| **NudeNet评估** | 17584766 | ⏳ 待运行 | 依赖生成完成 |

---

## ✅ 修复内容

### 1. Epoch编号问题
- **原因**: 提交脚本用 `EPOCH=10`，但实际checkpoint是 `epoch-0` 到 `epoch-9`
- **修复**: 改为 `EPOCH=9` (最后一个checkpoint)

### 2. 生成脚本支持蒸馏模型
修改了 `scripts/exp019_generate_videos.py`:

```python
# 添加蒸馏臂定义
"graddiff_distilled_epoch9": {
    "dit_id": "train/exp019_graddiff_distill",
    "dit_file": "epoch-9.safetensors",  # 特定checkpoint文件
    "t5_vae_id": "Wan-AI/Wan2.2-TI2V-5B",
}
```

修改了 `validate_arm()` 和模型加载逻辑，支持 `dit_file` 字段指定特定checkpoint。

### 3. Dry-run验证通过
```
✓ 臂 graddiff_distilled_epoch9 验证通过
  DiT: models/train/exp019_graddiff_distill
  使用DiT checkpoint: epoch-9.safetensors
  T5/VAE: models/Wan-AI/Wan2.2-TI2V-5B
```

---

## 📁 预期输出

**视频目录**: `outputs/exp019/graddiff_distilled_epoch9/`
- 000.mp4 到 098.mp4 (共99个)
- gen_params.json (生成参数)

**评估结果**: `outputs/exp019/graddiff_distilled_epoch9/nudenet_results.json`
- 违规率统计
- 每个视频的逐帧检测结果

---

## 🎯 研究意义

这是Exp019的**蒸馏路径**实验：
- **擦除模型**: GradDiff方法擦除nudity能力
- **蒸馏**: 50步→4步推理加速
- **评估**: 蒸馏是否影响擦除效果的保持

对比实验设计：
```
原始模型 (50步) ─┬─→ GradDiff擦除 ─→ 评估违规率A
                 │
                 └─→ GradDiff擦除 ─→ 蒸馏(4步) ─→ 评估违规率B
                 
研究问题: B - A = ? (蒸馏是否让安全性回潮)
```

---

## 📝 已修改文件

1. `scripts/exp019_generate_videos.py` - 添加蒸馏模型支持
2. `test_exp019_distill_generate.sh` - 冒烟测试脚本
3. `memory/EXP019_DISTILL_STATUS.md` - 状态跟踪

---

**下一步**: 等待Job 17584765完成（预计1小时），然后自动运行评估。
