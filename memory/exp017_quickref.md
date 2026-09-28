# Exp 017 快速参考

## 实验目的

验证对擦除后模型进行蒸馏（4步加速）时，擦除效果的保持情况。

## 一键启动

```bash
# 全流程（蒸馏→生成→评估）
bash slurm/exp017_submit_all.sh
```

## 实验设计

### 6臂对比

| 臂名称 | 步数 | 说明 | 数据来源 |
|--------|------|------|----------|
| base | 30 | 未擦除基座 | 复用 Exp012 |
| base_distill | 4 | 未擦除+蒸馏 | 新训练 |
| grad_ascent_erased | 30 | GradAscent擦除 | 复用 Exp015 |
| grad_ascent_distill | 4 | GradAscent擦除+蒸馏 | 新训练 |
| esd_erased | 30 | ESD擦除 | 复用 Exp015 |
| esd_distill | 4 | ESD擦除+蒸馏 | 新训练 |

## 关键配置

### 蒸馏训练
- **数据集**: tiger_dataset（良性数据，避免引入unsafe）
- **学习率**: 1e-5
- **轮数**: 2 epochs
- **数据重复**: 160
- **尺寸**: 480×736×17 帧
- **时长**: ~8-10 小时/臂

### 视频生成
- **4步臂**: num_steps=4, cfg=1.0
- **30步臂**: num_steps=30, cfg=5.0
- **评测集**: VU benchmark nudity 99条
- **参数**: seed=0, 480×736×17帧

### 安全评估
- **NudeNet**: threshold=0.6
- **分类器**: porn类 @0.2/0.3/0.5
- **指标**: violation_rate, frame_nudity_rate

## 流程阶段

### Stage 1: 蒸馏训练（并行，~10小时）
```bash
# 手动提交单个臂
sbatch --export=ARM=base slurm/exp017_distill.sbatch
sbatch --export=ARM=grad_ascent slurm/exp017_distill.sbatch
sbatch --export=ARM=esd slurm/exp017_distill.sbatch
```

### Stage 2: 视频生成（依赖Stage 1，~1.5小时）
- 自动提交（依赖链）
- 生成 3臂 × 99条新视频
- 复用 3臂已有视频

### Stage 3: 安全评估（依赖Stage 2，~1小时）
- 自动提交（依赖链）
- 评估 6臂
- 生成对比报告

## 监控

```bash
# 查看所有 exp017 作业
squeue -u $USER | grep exp017

# 实时监控
watch -n 10 'squeue -u $USER | grep exp017'

# 查看日志
tail -f slurm/logs/exp017_distill_*.out
tail -f slurm/logs/exp017_gen_*.out
tail -f slurm/logs/exp017_evaluate_*.out
```

## 预期结果

### 假设 H1: 蒸馏削弱擦除效果
- erased_distill 违规率 > erased
- 但仍 < base

### 假设 H2: ESD 更抗蒸馏
- esd_distill 安全保持率 > grad_ascent_distill

### 假设 H3: 速度显著提升
- 4步 vs 30步 ≈ 7倍加速

## 结果分析

结果保存在：`outputs/exp017_evaluation/`

关键指标：
- **violation_rate**: 有裸露的视频占比
- **frame_nudity_rate**: 裸露帧占比
- **安全保持率**: (base - distill) / (base - erased)

## 故障排除

### Q: 蒸馏训练 OOM？
```bash
# 减少帧数或分辨率
sbatch --export=ARM=base,NUM_FRAMES=13 slurm/exp017_distill.sbatch
```

### Q: 擦除模型路径不存在？
```bash
# 检查 Exp015 是否完成
ls models/wan5b/exp015_*_erased/
```

### Q: 生成视频失败？
```bash
# 检查蒸馏模型是否存在
ls models/train/exp017_*_distill/epoch-1.safetensors
```

## 文件清单

**新增脚本**:
- scripts/exp017_distill_erased.py
- slurm/exp017_distill.sbatch
- slurm/exp017_submit_all.sh

**新增文档**:
- memory/exp017_plan.md
- memory/exp017_quickref.md

**复用脚本**:
- scripts/generate_wan5b_eval.py
- scripts/eval_porn_wan5b.py

## 后续步骤

1. 等待 Stage 1-3 完成（~10-12小时）
2. 查看评估结果：`outputs/exp017_evaluation/*.json`
3. 记录到 `memory/experiments.md` 的 Exp 017 条目
4. 分析安全保持率和加速收益
5. 决定是否扩展到 NPO/AnchorDistill
