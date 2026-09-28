# Exp017 蒸馏评估补充 TodoList

**目标**: 评估 Exp017 已生成的 495 个蒸馏视频的安全性  
**时间**: ~30 分钟

---

## 📋 任务清单

### ✅ Task 1: 提交评估作业
```bash
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 提交评估任务（5个方法并行）
bash slurm/exp017_submit_eval.sh
```

**预期输出**:
```
✅ 已提交 5 个评估任务（array job）
Job ID: XXXXXXX
```

---

### ✅ Task 2: 监控评估进度
```bash
# 查看作业状态
squeue -j <JOB_ID>

# 查看日志（任选一个方法）
tail -f outputs/exp017_eval_distill_<JOB_ID>_0.log  # base
tail -f outputs/exp017_eval_distill_<JOB_ID>_1.log  # grad_ascent
tail -f outputs/exp017_eval_distill_<JOB_ID>_2.log  # esd
```

**预期时间**: ~30 分钟（5个方法并行）

---

### ✅ Task 3: 验证评估结果
```bash
# 检查生成的评估文件
ls -lh outputs/exp017_evaluation/

# 应该看到 5 个 JSON 文件：
# base_distill_evaluation.json
# grad_ascent_distill_evaluation.json
# esd_distill_evaluation.json
# npo_distill_evaluation.json
# anchor_distill_distill_evaluation.json
```

---

### ✅ Task 4: 分析蒸馏对安全性的影响
```bash
python3 << 'EOF'
import json

methods = ['base', 'grad_ascent', 'esd', 'npo', 'anchor_distill']

print("=== Exp017 蒸馏评估结果 ===\n")

# 加载 Exp015 的基线数据（50步推理）
exp015_data = {}
for method in methods:
    if method == 'base':
        file_path = 'outputs/exp015_evaluation/base_evaluation.json'
    else:
        file_path = f'outputs/exp015_evaluation/{method}_erased_evaluation.json'
    
    try:
        data = json.load(open(file_path))
        exp015_data[method] = data['nudenet']['summary']['violation_rate']
    except:
        exp015_data[method] = None

# 加载 Exp017 的蒸馏数据（4步推理）
exp017_data = {}
for method in methods:
    file_path = f'outputs/exp017_evaluation/{method}_distill_evaluation.json'
    try:
        data = json.load(open(file_path))
        exp017_data[method] = data['nudenet']['summary']['violation_rate']
    except:
        exp017_data[method] = None

# 对比表格
print("| 方法 | 50步推理 (Exp015) | 4步蒸馏 (Exp017) | 变化 |")
print("|------|-------------------|------------------|------|")
for method in methods:
    vr_50 = exp015_data.get(method)
    vr_4 = exp017_data.get(method)
    
    if vr_50 is not None and vr_4 is not None:
        change = (vr_4 - vr_50) * 100
        change_str = f"{change:+.1f}pp"
        print(f"| {method} | {vr_50:.1%} | {vr_4:.1%} | {change_str} |")
    else:
        print(f"| {method} | - | - | - |")

# 关键发现
print("\n=== 关键发现 ===")
if all(exp017_data.values()):
    avg_change = sum((exp017_data[m] - exp015_data[m]) * 100 for m in methods if exp015_data[m] is not None) / len(methods)
    print(f"平均变化: {avg_change:+.1f}pp")
    
    if avg_change > 5:
        print("⚠️  蒸馏显著降低了安全性")
    elif avg_change < -5:
        print("✅ 蒸馏意外改善了安全性")
    else:
        print("⚪ 蒸馏对安全性影响较小")

EOF
```

---

### ✅ Task 5: 更新 experiments.md
```bash
# 手动更新 memory/experiments.md 中 Exp017 的结果部分
# 填写违规率数据和关键发现
```

**更新内容**:
```markdown
- **Stage 3 评估** (2026-09-XX): ✅ 完成
  - Jobs: XXXXXXX (array 0-4)
  - 5个方法各评估 99 个视频
  
- **Results**: ✅ 完成
  - **蒸馏对安全性的影响**:
    | 方法 | 50步推理 | 4步蒸馏 | 变化 |
    |------|----------|---------|------|
    | base | XX.X% | XX.X% | XX.Xpp |
    | grad_ascent | XX.X% | XX.X% | XX.Xpp |
    | esd | XX.X% | XX.X% | XX.Xpp |
    | npo | XX.X% | XX.X% | XX.Xpp |
    | anchor_distill | XX.X% | XX.X% | XX.Xpp |
  
  - **关键发现**:
    - 蒸馏是否影响擦除效果？
    - 推理加速对安全性的影响方向？
```

---

### ✅ Task 6: 数据完整性检查
```bash
# 检查所有评估文件
python3 << 'EOF'
import json

methods = ['base', 'grad_ascent', 'esd', 'npo', 'anchor_distill']

print("=== 数据完整性检查 ===\n")

for method in methods:
    file_path = f'outputs/exp017_evaluation/{method}_distill_evaluation.json'
    try:
        data = json.load(open(file_path))
        total = data['nudenet']['summary']['total_videos']
        vr = data['nudenet']['summary']['violation_rate']
        
        if total == 99:
            print(f"✅ {method}: {total} videos, {vr:.1%}")
        else:
            print(f"⚠️  {method}: {total} videos (应该是 99)")
    except FileNotFoundError:
        print(f"❌ {method}: 文件不存在")
    except Exception as e:
        print(f"❌ {method}: 读取错误 - {e}")

EOF
```

---

## 📊 预期发现

### 假设 1: 蒸馏保持安全性
- 4步蒸馏的违规率与 50步推理相近（±2pp）
- 说明推理加速不影响擦除效果

### 假设 2: 蒸馏降低安全性
- 4步蒸馏的违规率显著升高（+5pp 以上）
- 说明加速过程中损失了安全对齐信息

### 假设 3: 蒸馏意外改善安全性
- 4步蒸馏的违规率降低（类似量化的意外发现）
- 可能是因为步数减少降低了生成细节的能力

---

## 🔗 相关文件

- 评估脚本: `slurm/exp017_eval_distill.sbatch`
- 提交脚本: `slurm/exp017_submit_eval.sh`
- 实验记录: `memory/experiments.md` (Exp017)
- 数据汇总: `memory/QUICK_DATA_REFERENCE.md`

---

**执行命令**:
```bash
bash slurm/exp017_submit_eval.sh
```
