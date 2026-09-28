# Exp015 错误记录

## 错误 #13: load_lora 参数名称错误

**时间**: 2026-09-03 10:59

**作业**: Jobs 17449669-17449676 (Stage 4 v2 全失败)

**现象**:
```
TypeError: BasePipeline.load_lora() got an unexpected keyword argument 'lora_alpha'
```

**根本原因**:
- 使用了错误的参数名 `lora_alpha=1.0`
- 本地 diffsynth 模块的正确参数名是 `alpha=1.0`
- 参考 `generate_wan5b_eval.py` 中的正确用法：`pipe.load_lora(pipe.dit, lora_path, alpha=1)`

**解决方案**:
修改 `scripts/exp015_generate_videos.py:237`：
```python
# 错误
pipe.load_lora(pipe.dit, arm["lora"], lora_alpha=1.0)

# 正确
pipe.load_lora(pipe.dit, arm["lora"], alpha=1.0)
```

**影响**:
- Stage 4 v2 的 8 个作业全部失败
- 重新提交为 v3: Jobs 17449895-17449902

**教训**:
- 参数名要严格按照本地 diffsynth 模块的 API
- 不要混用 DiffSynth-Studio 和本地 diffsynth 的参数名
- 参考成功案例的准确参数名（alpha vs lora_alpha）

---

# Exp015 错误记录

## 错误 #1: 环境配置错误 - diffsynth 缺少 diffusers

**时间**: 2026-08-31 08:03

**作业**: Job 17428673 (ESD merge), Job 17428674 (NPO merge)

**现象**:
```
ModuleNotFoundError: No module named 'diffusers'
```

**根本原因**:
- 合并脚本激活了 `conda activate diffsynth` 环境
- 该环境只有 torch，没有 diffusers 模块
- 需要使用 VU 项目的 `.venv` 环境

**解决方案**:
修改所有合并脚本的环境配置：
```bash
# 错误的配置
conda activate diffsynth

# 正确的配置
if [ -f "$VU_ROOT/.venv/bin/activate" ]; then
    source "$VU_ROOT/.venv/bin/activate"
else
    echo "[ERROR] 未找到 VU .venv 环境"
    exit 1
fi
```

**影响文件**:
- `slurm/exp015_merge_esd.sbatch` ✅ 已修复
- `slurm/exp015_merge_npo.sbatch` ✅ 已修复
- `slurm/exp015_merge_grad_ascent.sbatch` ✅ 已修复

**教训**:
- 不同阶段需要不同依赖：擦除训练需要 VU 依赖，合并需要 diffusers
- VU 项目使用 `.venv` 而非 conda 环境
- 应该在脚本中优先检查 `.venv`，fallback 到 conda

---

## 错误 #2: 环境名称错误 - video-unlearning 不存在

**时间**: 2026-08-31 08:16

**作业**: Job 17428681 (GradAscent v1)

**现象**:
```
EnvironmentNameNotFound: Could not find conda environment: video-unlearning
```

**根本原因**:
- GradAscent 脚本直接写了 `conda activate video-unlearning`
- 该 conda 环境不存在，VU 使用 `.venv`

**解决方案**:
使用与 ESD/NPO 相同的环境检测逻辑：
```bash
if [ -f "$VU_ROOT/.venv/bin/activate" ]; then
    source "$VU_ROOT/.venv/bin/activate"
elif conda env list | grep -q "video-unlearning"; then
    conda activate video-unlearning
else
    conda activate diffsynth
fi
```

**影响文件**:
- `slurm/exp015_unlearn_grad_ascent.sbatch` ✅ 已修复

---

## 错误 #3: AnchorDistill 概念词检查失败

**时间**: 2026-08-31 07:52

**作业**: Job 17428670 (AnchorDistill v1)

**现象**:
```
ValueError: forget prompt lacks erase concept 'nudity', so no destination prompt can be built
```

**根本原因**:
- 使用了非空 anchor (`anchor="a person"`)
- 触发了 AnchorDistill 的概念词检查（line 66-72）
- VU 的 nudity 数据集使用描述性 prompt（"bare breasts clearly visible"）
- Prompt 中不包含 "nudity" 这个概念词

**AnchorDistill 代码逻辑**:
```python
if self.anchor:  # 只在非空 anchor 时检查
    for example in self.examples:
        if self.erase_concept not in example["prompt"]:
            raise ValueError(f"forget prompt lacks erase concept '{self.erase_concept}'...")
```

**VU 项目的解决方案**:
- 在 text_encoder 位置使用 `anchor=""` (blank distillation)
- 空 anchor 绕过概念词检查
- 目标 prompt 变为空字符串，而非替换后的 prompt

**我们的解决方案**:
- 在 transformer 位置使用 `anchor=""` (与 VU 的 encoder 位置形成对比)
- 修改脚本传入 `--anchor ""`

**影响文件**:
- `slurm/exp015_unlearn_anchor_distill.sbatch` ✅ 已修复 (Job 17428692)

**教训**:
- 仔细阅读方法实现的条件检查逻辑
- 数据集格式会限制某些方法的适用性
- VU 项目的实验记录是宝贵参考（exp503 显示他们用了 enc_distill_empty）

---

## 错误 #4: GradDiff 方法不支持

**时间**: 2026-08-31 07:52

**作业**: Job 17428667 (GradDiff)

**现象**:
```
error: invalid choice: 'GradDiff' (choose from GradAscent, AnchorDistill, ESD, NPO)
```

**根本原因**:
- `run_unlearn_wan5b.py` 的 argparse choices 只有 4 个方法
- GradDiff 未实现 Wan5B 配置文件

**解决方案**:
- 跳过 GradDiff（需要额外实现工作）
- 聚焦于 4 种已支持的方法

**影响**:
- Exp015 从 5 种方法降为 4 种
- 仍然足够对比（GradAscent 是 GradDiff 的 forget-only 版本）

---

## 错误 #5: 模板变量未替换

**时间**: 2026-08-31 07:51（第一次提交时）

**作业**: 所有擦除训练脚本

**现象**:
所有脚本都显示 `METHOD="grad_diff"` 而非实际方法名

**根本原因**:
- 脚本从模板复制时，变量占位符未替换
- `METHOD="${METHOD:-grad_diff}"` 应该改为具体方法名

**解决方案**:
每个脚本显式设置方法名：
```bash
METHOD="esd"  # 或 npo, grad_ascent, anchor_distill
```

**影响文件**:
- 所有 `exp015_unlearn_*.sbatch` 脚本 ✅ 已修复

**教训**:
- 模板化脚本需要明确替换点
- 使用 sed 批量替换时要验证结果

---

## 错误 #6: 合并脚本参数错误

**时间**: 2026-08-31 08:19

**作业**: Job 17428686 (ESD merge v2), Job 17428687 (NPO merge v2)

**现象**:
```
merge_unlearn_lora_wan5b.py: error: the following arguments are required: --adapter, --base-dir
```

**根本原因**:
- 脚本使用了错误的参数名：`--base-model`、`--lora-dir`、`--lora-alpha`、`--lora-rank`
- 实际参数是：`--adapter`、`--base-dir`、`--output-dir`、`--bridge-output-dir`
- 没有先验证脚本接口就直接写了参数

**解决方案**:
修正参数并简化流程（合并+桥转换一步完成）：
```bash
python3 scripts/merge_unlearn_lora_wan5b.py \
    --adapter "$UNLEARN_LORA_DIR/adapter_final.pt" \
    --base-dir "$BASE_DIFFUSERS/transformer" \
    --output-dir "$OUTPUT_ERASED_DIFFUSERS/transformer" \
    --bridge-output-dir "$OUTPUT_ERASED_DIFFSYNTH" \
    --force
```

**影响文件**:
- `slurm/exp015_merge_esd.sbatch` ✅ 已修复 (重新提交)
- `slurm/exp015_merge_npo.sbatch` ✅ 已修复 (重新提交)
- `slurm/exp015_merge_grad_ascent.sbatch` ✅ 已修复 (重新提交)

**教训**:
- 调用脚本前先 `--help` 确认参数接口
- Exp012 的脚本可能已更新，不能假设参数名不变
- 第一次失败后应该立即检查 --help 而非反复重试

---

## 总结

**共计 8 个错误**:
1. ✅ 环境配置错误（diffsynth 缺 diffusers）
2. ✅ 环境名称错误（video-unlearning 不存在）
3. ✅ AnchorDistill 概念词检查失败
4. ⚠️ GradDiff 方法不支持（跳过）
5. ✅ 模板变量未替换
6. ✅ 合并脚本参数错误
7. ✅ 微调脚本数据集路径错误
8. ✅ 微调环境 torch/transformers 版本不兼容

**根本原因分类**:
- 环境配置: 4 个（#1, #2, #8, 部分 #5）
- 接口理解: 2 个（#3, #6）
- 路径配置: 1 个（#7）
- 方法支持: 1 个（#4）

**改进措施**:
1. 创建统一的环境检测函数
2. 在提交前运行 prerequisite checker
3. 先在单个方法上验证，再批量展开
4. 仔细阅读 VU 项目的实验记录和方法实现

---

## 错误 #8: 微调环境 torch/transformers 版本不兼容

**时间**: 2026-08-31 11:30

**作业**: Jobs 17429374-17429377 (所有微调作业 v2)

**现象**:
```
ImportError: cannot import name 'NP_SUPPORTED_MODULES' from 'torch._dynamo.utils'
```

**根本原因**:
- 使用了 `diffsynth` conda 环境
- 该环境的 `transformers==5.14.1` 版本异常（不存在的未来版本）
- 与 `torch==2.13.0` 不兼容，导致 import 失败
- Exp012 成功案例使用的是 **VU .venv 环境**

**解决方案**:
修改所有微调脚本使用 VU .venv 环境（与擦除/合并一致）：
```bash
VU_ROOT="/home/x_jiage/jiage/video-unlearning"
if [ -f "$VU_ROOT/.venv/bin/activate" ]; then
    source "$VU_ROOT/.venv/bin/activate"
fi
```

**影响文件**:
- `slurm/exp015_finetune_esd.sbatch` ✅ 已修复 (重新提交 Job 17429530)
- `slurm/exp015_finetune_npo.sbatch` ✅ 已修复 (重新提交 Job 17429531)
- `slurm/exp015_finetune_grad_ascent.sbatch` ✅ 已修复 (重新提交 Job 17429532)
- `slurm/exp015_finetune_anchor_distill.sbatch` ✅ 已修复 (重新提交 Job 17429533)

**教训**:
- 同一实验的所有阶段应使用统一环境（Exp015: Stage 1/2 用 VU .venv，Stage 3 应该也用）
- 参考成功案例的完整配置，而非仅复制脚本片段
- 版本异常（如 transformers 5.14.1）提示环境可能损坏

---

## 错误 #9: 微调脚本参数名错误

**时间**: 2026-08-31 11:36

**作业**: Jobs 17429530-17429533 (微调作业 v3)

**现象**:
```
train.py: error: the following arguments are required: --dataset_base_path
```

**根本原因**:
- 使用了错误的参数名 `--dataset_path`
- `train.py` 实际需要 `--dataset_base_path` + `--dataset_metadata_path`（分离目录和文件）
- 缺少必需参数：`--remove_prefix_in_ckpt`, `--lora_base_model`
- 参数名错误：`--lora_targets` 应为 `--lora_target_modules`

**解决方案**:
参考 Exp012 成功案例，使用正确的参数集：
```bash
accelerate launch examples/wanvideo/model_training/train.py \
    --dataset_base_path "data/tiger_dataset" \
    --dataset_metadata_path "data/tiger_dataset/metadata_100.csv" \
    --remove_prefix_in_ckpt pipe.dit. \
    --lora_base_model dit \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 32 \
    ...
```

**影响文件**:
- 所有微调脚本 ✅ 已修复 (重新提交 Jobs 17429704-17429707)

**教训**:
- 直接调用 accelerate 时必须先 `--help` 验证所有参数
- 复制成功案例的完整命令，而非自行推测参数名
- 第一次失败就应该对比成功案例，而非反复试错

---

## 改进措施总结

**环境管理**:
1. 同一实验的所有阶段应使用统一环境
2. 优先使用 VU .venv（完整依赖）而非 diffsynth conda

**参数验证**:
1. 提交前运行 `--help` 验证参数接口
2. 参考成功案例的完整配置
3. 使用路径检查逻辑捕获早期错误

**错误处理**:
1. 第一次失败立即诊断根因，而非反复重试
2. 记录每个错误的详细信息到 memory/exp015_errors.md
3. 归类错误原因，识别系统性问题

---

## 错误 #10: accelerate 参数传递给 train.py

**时间**: 2026-08-31 11:43

**作业**: Jobs 17429704-17429707 (微调作业 v4)

**现象**:
```
train.py: error: unrecognized arguments: --dataloader_num_workers 4 --mixed_precision bf16 --train_batch_size 1 --max_grad_norm 1.0
```

**根本原因**:
- 将 accelerate 层面的配置参数（`--dataloader_num_workers`, `--mixed_precision`, `--train_batch_size`, `--max_grad_norm`）直接传递给了 `train.py`
- `train.py` 只接受模型训练相关的参数，不接受 accelerate 框架配置参数
- 这些参数应该通过 accelerate 配置文件或在 `accelerate launch` 命令行中设置，而非传递给训练脚本

**正确做法**（参考 `scripts/finetune_wan5b.py`）:
```bash
# 只传递 train.py 接受的参数
accelerate launch examples/wanvideo/model_training/train.py \
    --dataset_base_path "data/tiger_dataset" \
    --dataset_metadata_path "$DATASET_PATH" \
    --height 480 --width 832 --num_frames 25 \
    --model_id_with_origin_paths "..." \
    --learning_rate 1e-4 \
    --num_epochs 2 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0.01 \
    --output_path "$OUTPUT_DIR" \
    --remove_prefix_in_ckpt pipe.dit. \
    --lora_base_model dit \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --lora_rank 32 \
    --use_gradient_checkpointing \
    --enable_tensorboard_log
# 注意：没有 --dataloader_num_workers, --mixed_precision, --train_batch_size, --max_grad_norm
```

**影响文件**:
- `slurm/exp015_finetune_esd.sbatch` ✅ 已修复
- `slurm/exp015_finetune_npo.sbatch` ✅ 已修复
- `slurm/exp015_finetune_grad_ascent.sbatch` ✅ 已修复
- `slurm/exp015_finetune_anchor_distill.sbatch` ✅ 已修复

**教训**:
- 区分两层参数：accelerate 框架配置 vs 训练脚本参数
- 参考成功案例的完整命令结构，而非仅复制参数列表
- 第一次失败后应该立即对比成功案例，而非累积多次失败

**根因分类**: 接口理解错误（accelerate 参数传递机制）

---

## 错误 #11: model_id_with_origin_paths 格式错误

**时间**: 2026-08-31 11:48

**作业**: Jobs 17429720-17429723 (微调作业 v5)

**现象**:
```
Loading models from: []
ValueError: Cannot detect the model type. File: []. Model hash: d41d8cd98f00b204e9800998ecf8427e
```

**根本原因**:
1. **分隔符错误**: 使用分号 `;` 而非逗号 `,` 分隔多个文件
2. **格式错误**: 未对每个文件单独指定 `model_id:pattern`

错误格式：
```bash
exp015_esd_erased:diffusion_pytorch_model*.safetensors;models_t5_umt5-xxl-enc-bf16.safetensors;Wan2.2_VAE.safetensors
```

正确格式（参考 `scripts/finetune_wan5b.py`）：
```bash
wan5b/exp015_esd_erased:diffusion_pytorch_model*.safetensors,wan5b/exp015_esd_erased:models_t5_umt5-xxl-enc-bf16.safetensors,wan5b/exp015_esd_erased:Wan2.2_VAE.safetensors
```

**解决方案**:
```bash
MODEL_ID="wan5b/exp015_${METHOD}_erased"
MODEL_PATHS="${MODEL_ID}:diffusion_pytorch_model*.safetensors,${MODEL_ID}:models_t5_umt5-xxl-enc-bf16.safetensors,${MODEL_ID}:Wan2.2_VAE.safetensors"
accelerate launch ... --model_id_with_origin_paths "${MODEL_PATHS}"
```

**影响文件**:
- 所有微调脚本 ✅ 已修复

**教训**:
- 复杂参数格式应先用 `--dry-run` 验证
- 第一次模型加载失败就应该对比成功案例的完整命令
- 不要假设分隔符，严格按照文档/成功案例的格式

**根因分类**: 参数格式错误（model_id_with_origin_paths 语法）

---

## 错误 #12: 导入错误 - 使用了错误的 DiffSynth API

**时间**: 2026-08-31

**作业**: Jobs 17449135-17449142 (Stage 4 视频生成 v1)

**现象**:
```
ImportError: cannot import name 'ModelManager' from 'diffsynth'
```

**根本原因**:
- 脚本使用了 DiffSynth-Studio 包的 API（`ModelManager`, `WanVideoPipeline.from_model_manager`）
- 但 FT 项目使用的是本地 `diffsynth` 模块，API 不同
- 应该使用 `WanVideoPipeline.from_pretrained` + `ModelConfig`

**正确用法**（参考 `scripts/generate_wan5b_eval.py`）:
```python
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video

model_configs = [
    ModelConfig(model_id=arm['model_id'], origin_file_pattern=pattern)
    for pattern in MODEL_FILES
]
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=model_configs,
    tokenizer_config=tokenizer_config,
)
# 生成
video = pipe(prompt=..., cfg_scale=..., sigma_shift=..., tiled=True)
save_video(video, path, fps=...)
```

**修复**:
- ✅ 更新导入和模型加载代码
- ✅ 修正生成参数名（`cfg_scale` vs `guidance_scale`，`sigma_shift` vs `flow_shift`）
- ✅ 添加 `tiled=True` 参数
- ✅ 使用 `save_video` 函数而非 `pipe.save_video`

**教训**:
- 不要假设 API，先查看同项目内的成功案例
- 第一次导入失败就应该立即检查参考脚本
- 本地模块和包的 API 可能完全不同

**根因分类**: API 使用错误（本地 diffsynth vs DiffSynth-Studio 包）
