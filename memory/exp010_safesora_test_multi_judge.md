# Exp 010 — SafeSora 测试集三方判别器 vs 人工标注

> 位置：项目内 `memory/exp010_safesora_test_multi_judge.md`
> GPU 申请说明：见项目内共享文档 `memory/gpu_node_apply.md`

## 1. 实验需求

在 SafeSora-Label 官方 **test 集**（5745 条 text-video 对，带**人工标注**）上，用
三个独立判别器（分类器 best.pt 三阈值、Qwen3-VL-8B、GPT-4o-mini）对每个视频做
safe/unsafe 二分类，与**人工标注 is_safe** 对照，评估各判别器可靠性，并量化
判别器之间的一致性，从而验证"安全分类器是否可信"。

为什么用 SafeSora test 集而非生成的 base/lora/... 视频（Exp 009）：
- Exp 009 的 base/lora/... 是 **prompt 生成**的视频，**没有人工视频级标注**，
  之前误用 prompt 标签当 GT 得出低 F1，那是"prompt unsafe 但生成视频不 unsafe"，
  不能用于评估分类器可靠性。
- SafeSora-Label test 集有**人工整体判断 `is_safe`**（4878 safe / 867 unsafe，
  较均衡 15% unsafe），是可信的 ground truth。

## 2. 数据准备

- 标注: `/home/x_jiage/jiage/datasets/SafeSora-Label/config-test.json.gz` (5745 条)
  - key 字段: `video_id`, `video_path`, `is_safe`(人工整体二元), `video_labels`(12类),
    `prompt_labels`(prompt 级 12 类), `prompt_type`
- 视频: `/home/x_jiage/jiage/datasets/SafeSora/videos/` (57333 个，含 train)
- 构造 test 视频软链目录 `outputs/safesora_test_videos/`（仅 5745 个 test 视频，
  文件名 = `<video_id>.mp4`），使三个判别器可统一消费。

```python
import gzip, json, os
src_root='.../SafeSora/videos'; dst_root='outputs/safesora_test_videos'
# 遍历 config-test.json.gz, os.symlink(src, dst) 按 video_id
```

判键：判别器结果用 `os.path.basename(video)`（= `<video_id>.mp4`）作 key，
compare_with_gt 的 GT 匹配键 = `video_id`（basename 去 .mp4）。

## 3. 判别器

| 判别器 | 实现 | 帧数 | 资源 |
|--------|------|------|------|
| cls_thr_0.20/0.30/0.50 | `batch_predict_dir.py` + best.pt | 8 | A100 1卡, 约 10min/阈值 |
| qwen3_vl | `qwen_vl_safety_judge.py` + Qwen3-VL-8B | 8 | A100 1卡, 约 3.5h |
| gpt4o | `gpt4o_safety_judge.py` + gpt-4o-mini | 4 | berzelius 强制 GPU 1卡 + API, 约 3h |

三阈值对应 Exp 阈值优化：0.5=best_acc, 0.3=中间, 0.2=best_f1。

输出: `outputs/safesora_test_multi_judge/test/<judge>.json`

## 4. 二分类口径

- GT: `binary_unsafe = not is_safe`
- 判别器: `pred_unsafe`（分类器=任一 unsafe 类命中；qwen/gpt4o=模型 FINAL 判断）
- 无 pred_unsafe 或 error/parse_failed → 该项排除
- 指标: TP/FP/TN/FN + acc/prec/rec/F1（unsafe=positive）

## 5. sbatch 脚本

- `slurm/safesora_test_cls.sbatch` — 分类器三阈值
- `slurm/safesora_test_qwen.sbatch` — Qwen3-VL（含 buildenv-gcccuda 加载, 见 Exp009 教训）
- `slurm/safesora_test_gpt4o.sbatch` — GPT-4o（含 API smoke test; berzelius 强制 --gres=gpu:1）

作业:
- 分类器: 17339382 (14:10-14:32, EXIT 0)
- Qwen: 17339383 (14:10-16:27, EXIT 0)
- GPT-4o 首次: 17339384 (14:10-15:21, **被集群 CANCELLED** 于 3040/5745，
  因脚本当时只在末尾一次性写盘, 3040 条结果丢失)
- GPT-4o 二次: 17343164 (22:15 重提, 已改进定期持久化)

## 6. 结果（截至分类器+Qwen 完成；GPT-4o 运行中）

### 6.1 unsafe_rate vs GT
GT unsafe = 867/5745 = 15.1%
- cls_0.20: 32.6%
- cls_0.30: ~25%
- cls_0.50: 21.9% (1175/5745 附近)
- qwen3_vl: 15.0% (与 GT 几乎吻合)
- gpt4o: Pending（早期进度 ~27.5%）

### 6.2 二分类指标 (5743-5745 条)
| judge | n | acc | prec | rec | F1 |
|-------|---|-----|------|-----|-----|
| cls_0.20 | 5745 | 0.800 | 0.425 | 0.919 | 0.581 |
| cls_0.30 | 5745 | 0.830 | 0.467 | 0.874 | 0.608 |
| cls_0.50 | 5745 | 0.871 | 0.550 | 0.776 | **0.644** |
| qwen3_vl | 5743 | **0.893** | **0.644** | 0.647 | **0.645** |
| gpt4o | Pending | | | | |

### 6.3 初步结论
1. 分类器 (thr0.5) 与 Qwen3-VL **F1 几乎相同** (0.644 vs 0.645)，acc 略低但 rec 高。
   说明**安全分类器在人工标注测试集上是可靠的**，性能与顶级 VLM 相当。
2. Qwen3-VL unsafe_rate=15.0% 与 GT=15.1% **几乎精确吻合**；分类器 thr0.5 略高估(21.9%)。
3. 阈值 0.5 依然最优（F1=0.644 > 0.608 > 0.581），验证 Exp 阈值优化选择。
4. 这**否定了 Exp 009 的"分类器低 F1"假象**——那是用 prompt 标签误当视频 GT 所致。
5. GPT-4o 补全后对比三方一致性与各自指标。

## 7. 运行命令（SOP）

```bash
# 分类器三阈值
sbatch slurm/safesora_test_cls.sbatch
# Qwen3-VL（gcc buildenv）
sbatch slurm/safesora_test_qwen.sbatch
# GPT-4o
sbatch slurm/safesora_test_gpt4o.sbatch

# 对比报告 (is_safe 口径)
python -u -m classify.evaluation.compare_with_gt \
  --root outputs/safesora_test_multi_judge \
  --methods test \
  --judges "cls_thr_0.20,cls_thr_0.30,cls_thr_0.50,qwen3_vl,gpt4o" \
  --gt_format safesora_is_safe \
  --out_dir outputs/safesora_test_multi_judge/gt_comparison
```

## 8. 交付物
- `outputs/safesora_test_multi_judge/test/<judge>.json`（每视频 pred_probs/pred_unsafe）
- `outputs/safesora_test_multi_judge/gt_comparison/{per_judge_metrics.json, all_judges_vs_gt.json, mislabel_candidates.json, report.md}`
- 本文档 + `memory/experiments.md` Exp 010 记录

## 9. 注意事项 / 教训
- berzelius partition 强制 GPU，GPT-4o 也须 `--gres=gpu:1`（即使只需 API）。
- **断点续跑必须定期持久化**：GPT-4o 首次作业被集群 CANCELLED 丢 3040 条结果
  （脚本原只在末尾写盘）。已在 `gpt4o_safety_judge.py` 加 `save_progress()` 每 60s
  原子写盘（tmp+os.replace），被杀后可续跑。
- qwen 已实现持久化；cls 快不需。