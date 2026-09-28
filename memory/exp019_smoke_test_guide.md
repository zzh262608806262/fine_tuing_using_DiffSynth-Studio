# Exp019 冒烟测试与提交指南

**当前状态**: 在交互式节点 node051 (Job 17530953) 上进行冒烟测试

---

## 🧪 冒烟测试步骤

### 在交互式节点上运行：

```bash
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 运行冒烟测试（10步，约1-2分钟）
bash scripts/exp019_smoke_test.sh
```

冒烟测试将验证：
1. ✅ VU 环境正确激活
2. ✅ Python 依赖完整
3. ✅ 关键文件存在
4. ✅ GradDiff 训练能正常运行
5. ✅ Checkpoint 正常生成

---

## 🚀 冒烟测试通过后

### 方法 1: 提交单个擦除训练作业

```bash
sbatch slurm/exp019_unlearn_graddiff.sbatch
```

### 方法 2: 提交完整流程链（推荐）

```bash
bash scripts/exp019_submit_chain.sh
```

这将自动提交全流程：
- ✅ 擦除训练（600步）
- ✅ LoRA 合并（base + erased）
- ✅ 基线生成（2臂×99视频）
- ✅ 基线评估（NudeNet）
- ✅ 微调训练（repeat=25, epochs=20）
- ✅ 微调后生成（99视频）
- ✅ 微调后评估（NudeNet）

所有作业通过 SLURM 依赖链自动编排，无需手动干预。

---

## 📊 监控命令

```bash
# 查看所有作业
squeue -u $USER

# 运行进度监控
bash scripts/exp019_monitor.sh

# 查看特定作业日志
tail -f outputs/exp019_*.log
```

---

## ⚠️ 如果冒烟测试失败

1. **环境问题**: 检查 VU .venv 环境是否正确
   ```bash
   ls -lh /home/x_jiage/jiage/video-unlearning/.venv/bin/activate
   ```

2. **文件缺失**: 检查模型和数据是否存在
   ```bash
   ls models/Wan-AI/Wan2.2-TI2V-5B/model_index.json
   ls data/wan5b/unlearn_baseline/latent_manifest.jsonl
   ```

3. **CUDA 问题**: 确认 GPU 可用
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## 🎯 预期结果

冒烟测试成功后，将看到：
- ✅ `models/unlearn/exp019_smoke_test/step-10/` 目录
- ✅ Checkpoint 文件（约 153M）
- ✅ 训练日志无错误

全量训练（600步）预计：
- ⏱️ 时间: ~10 分钟
- 📁 输出: 7 个 checkpoints (step-100 到 step-600)
- 💾 大小: 每个约 153M

---

**祝测试顺利！** 🚀
