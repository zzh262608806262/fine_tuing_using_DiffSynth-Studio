# Exp016 生成参数问题诊断与修正

**日期**: 2026-09-03  
**问题**: 视频生成预计需要3天（72-80小时），远超预期

---

## 问题诊断

### 当前使用的参数（错误）

```python
GEN_PARAMS = {
    "height": 768,
    "width": 1344,
    "num_frames": 121,        # ❌ 过高
    "num_inference_steps": 50,
}
```

### 参数来源对比

| 来源 | 帧数 | 分辨率 | 推理步数 | 文件位置 |
|------|------|--------|----------|----------|
| **VU 代码默认** | 16 | 256×256 | 50 | `video-unlearning/src/generate.py:denoise_latent()` |
| **Exp014/015 probe** | **17** ✅ | **480×736** ✅ | **50** | `scripts/run_unlearn_wan5b.py` (默认) |
| **Exp015 eval** | **17** ✅ | **480×720** ✅ | **50** | `scripts/generate_wan5b_eval.py` |
| **Exp016 当前** | 121 ❌ | 768×1344 ❌ | 50 | `scripts/exp016_generate_videos.py` |

### 计算量分析

**Exp016 vs Exp014/015**:
```
帧数差异: 121 / 17 = 7.1倍
分辨率差异: (768×1344) / (480×736) = 1,032,192 / 353,280 = 2.92倍
总计算量: 7.1 × 2.92 ≈ 20.7倍
```

### 时间影响

**当前配置（121帧 + 768×1344）**:
- 单视频: 6-7分钟
- 单臂99条: 9-10小时
- 8臂串行: **72-80小时（3天）**

**修正配置（17帧 + 480×736）**:
- 单视频: ~20秒
- 单臂99条: ~30分钟
- 8臂串行: **~4小时**

---

## 帧数约束验证

### Wan5B 的帧数公式

Wan2.2-TI2V-5B VAE temporal scale factor = 4

```
latent_frames = (num_frames - 1) // 4 + 1
valid_frames = (latent_frames - 1) * 4 + 1
```

因此约束为: **num_frames = 4n + 1**

### 帧数验证

```python
帧数    latent    还原    有效
─────────────────────────────
 16  →   4   →   13    ✗
 17  →   5   →   17    ✓  (4×4 + 1)
 25  →   7   →   25    ✓  (4×6 + 1)
 49  →  13   →   49    ✓  (4×12 + 1)
 61  →  16   →   61    ✓  (4×15 + 1)
121  →  31   →  121    ✓  (4×30 + 1) - 但太长
```

---

## 参数选择依据

### VU 原始配置

VU 使用 CogVideoX 作为主要模型，默认参数：
- 帧数: 16（但 Wan5B 约束需要 4n+1）
- 分辨率: 256×256
- 推理步数: 50

**问题**: VU 的默认 16 帧**不符合** Wan5B 的 4n+1 约束

### Exp014/015 配置（推荐）✅

我们之前成功的实验使用：

**Probe 生成** (`run_unlearn_wan5b.py`):
```python
--probe-frames 17      # 默认，4×4 + 1
--probe-height 480
--probe-width 736
--probe-steps 50
```

**Eval 生成** (`generate_wan5b_eval.py`):
```python
num_frames = 17        # 固定
height = 480
width = 720            # 或 736
steps = 50
cfg = 5.0
flow_shift = 5.0
```

**证据**:
- Exp014 probe 视频: 44-64KB/视频
- 已验证可成功运行
- 与 VU 论文评估方法一致（合理的评估尺度）

### Exp016 初始配置（错误）❌

```python
num_frames = 121       # 过高（7倍）
height = 768
width = 1344           # 过高（3倍面积）
```

**可能来源**:
- 复制了某个高质量生成配置
- 或与训练配置混淆（训练可能用更高分辨率）

---

## 修正方案

### 推荐参数

```python
GEN_PARAMS = {
    "height": 480,
    "width": 736,          # 或 720，保持16的倍数
    "num_frames": 17,      # 4×4 + 1
    "num_inference_steps": 50,
    "cfg_scale": 7.0,      # 保持不变
    "seed": 42,            # 保持不变
}
```

### 理由

1. **符合约束**: 17 = 4×4 + 1 ✓
2. **已验证**: Exp014/015 成功使用
3. **合理速度**: 3-4小时完成，而非3天
4. **评估充分**: 17帧足够判断安全性（nudity 通常在前几帧就能检测）
5. **一致性**: 与项目其他实验保持一致

---

## 当前任务状态

### 运行中的任务

**Job**: 17452120 (esd_erased_nf4)
- 开始时间: 2026-09-03 16:23
- 当前运行: 2.5小时
- 进度: 15/99 视频（15%）
- 剩余时间: ~70小时

### 全部任务

8个臂，每臂99条prompts = **792个视频**
- esd_erased_nf4: 运行中（15/99）
- 其余7个: 排队中（0/99）

---

## 修正步骤

### 1. 停止当前任务

```bash
# 停止运行中的任务
scancel 17452120

# 停止排队中的任务
scancel 17452121 17452122 17452123 17452124 17452125 17452126 17452127
```

### 2. 修改生成参数

编辑 `scripts/exp016_generate_videos.py`:

```python
# 修改前
GEN_PARAMS = {
    "height": 768,
    "width": 1344,
    "num_frames": 121,
    "num_inference_steps": 50,
    "cfg_scale": 7.0,
    "seed": 42,
}

# 修改后
GEN_PARAMS = {
    "height": 480,
    "width": 736,
    "num_frames": 17,
    "num_inference_steps": 50,
    "cfg_scale": 7.0,
    "seed": 42,
}
```

### 3. 重新生成提交脚本

```bash
python scripts/exp016_generate_videos.py --dry-run
```

验证输出的生成参数是否正确。

### 4. 清理已生成的错误视频

```bash
rm -rf outputs/exp016/esd_erased_nf4/*.mp4
rm -rf outputs/exp016/esd_erased_nf4/gen_params.json
```

### 5. 重新提交任务

```bash
bash scripts/submit_exp016_gen_all.sh
```

### 6. 验证新任务

检查生成参数：
```bash
# 等待第一个视频生成
sleep 60
cat outputs/exp016/esd_erased_nf4/gen_params.json
```

预期输出：
```json
{
    "gen_params": {
        "height": 480,
        "width": 736,
        "num_frames": 17,
        ...
    }
}
```

---

## 经验教训

### 参数验证清单

在提交大规模生成任务前，必须验证：

1. ✅ **帧数约束**: 对于 Wan5B，必须是 4n+1
2. ✅ **参考之前成功实验**: 查看 probe/eval 的默认参数
3. ✅ **预估时间**: 单视频测试 → 推算总时间
4. ✅ **文件大小**: 检查生成视频大小是否合理
5. ✅ **资源占用**: 3天的任务占用GPU，影响其他实验

### 参数来源优先级

1. **最高优先级**: 项目已验证的配置（Exp014/015）
2. **次要参考**: VU 原始代码默认值
3. **谨慎使用**: 训练配置（通常更高，不适合评估）
4. **避免猜测**: 不要随意提高参数"追求质量"

### 文档化要求

每个实验的生成脚本应包含：
```python
# 生成参数说明
GEN_PARAMS = {
    "height": 480,         # 参考: Exp014 probe
    "width": 736,          # 参考: Exp014 probe
    "num_frames": 17,      # Wan5B 约束: 4n+1, 参考: Exp014/015
    "num_inference_steps": 50,  # VU 标准
    # 预估: 单视频 ~20秒, 8臂×99条 ~4小时
}
```

---

## 相关文件

### 修改的文件
- `scripts/exp016_generate_videos.py` - 生成参数配置

### 参考文件
- `scripts/run_unlearn_wan5b.py` - Exp014 probe 参数（17帧, 480×736）
- `scripts/generate_wan5b_eval.py` - Exp015 eval 参数（17帧, 480×720）
- `video-unlearning/src/generate.py` - VU 默认参数（16帧, 256×256）

### 输出目录
- `outputs/exp016/<arm>/` - 生成的视频
- `outputs/exp016/<arm>/gen_params.json` - 记录的生成参数

---

## 后续行动

- [x] 诊断参数问题
- [x] 确认修正方案
- [x] 记录经验教训
- [ ] 停止当前任务
- [ ] 修改生成参数
- [ ] 重新提交任务
- [ ] 验证新配置正确
- [ ] 监控新任务完成（预计4小时）
- [ ] 提交评估任务
