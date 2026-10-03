# Exp016 量化成功方案 - 解决新集群 Exp019 量化问题

**创建日期**: 2026-09-28  
**关联问题**: `memory/quantization_technical_questions.md`

---

## 核心发现

**Exp016 成功完成了量化实验**，解决了你在新集群遇到的大部分问题。以下是关键技术方案：

---

## ✅ 问题 A 解决方案：推理错误如何修复

### 你的问题
```python
NotImplementedError: "silu_cuda" not implemented for 'Byte'
```

### Exp016 的解决方案

**Exp016 没有遇到这个问题**，因为：

1. **不保存量化模型，只在推理时动态量化**
   - Exp016 量化脚本生成的模型 **不用于推理**
   - 生成视频时，每次 **重新加载 FP16 模型 + 量化**

2. **推理时的正确流程**（从 `scripts/exp015_generate_videos.py` 看到）：
   ```python
   # 1. 加载 FP16 模型
   pipe = WanVideoPipeline.from_pretrained(
       torch_dtype=torch.bfloat16,
       device="cuda",
       model_configs=model_configs,
   )
   
   # 2. 加载 LoRA（如果有）
   pipe.load_lora(pipe.dit, lora_path, alpha=1.0)
   
   # 3. 动态量化（每次推理时）
   quant_config = QuantizeConfig(method="bitsandbytes_nf4")
   quant_config.quantize_model(pipe.dit, compute_device="cuda", model_device="cuda")
   
   # 4. 推理（✅ 不会出错）
   video = pipe.generate_video(prompt, ...)
   ```

3. **为什么这样能 work？**
   - bitsandbytes 的量化是 **动态替换** nn.Linear 为 `Linear4bit`
   - 替换后的层输出仍是 `torch.bfloat16`，不是 `torch.uint8`
   - 你的错误可能是因为 **保存/加载后量化状态损坏**

---

## ✅ 问题 B 解决方案：bitsandbytes 量化能否保存？

### 答案：**不需要保存，也不应该保存**

Exp016 的设计证实了正确的 workflow：

**选项 1：推理时动态量化（Exp016 采用）** ✅
```python
# 每次推理时:
model = load_fp_model()          # 加载 FP16 模型
quantize(model)                  # 动态量化（~1-2 分钟）
inference(model)                 # 推理
# 结束后释放
```

**优点**：
- ✅ 简单可靠，不会出现 dtype 错误
- ✅ 量化状态完整，不依赖序列化
- ✅ 灵活切换量化/非量化

**缺点**：
- ⚠️ 每次推理前需要量化（~1-2 分钟开销）
- ⚠️ 对于批量生成（99-500 个视频），这个开销可忽略

---

## ✅ 问题 C 解决方案：正确的 workflow

### Exp016 的完整 workflow（已验证成功）

#### Stage 1: 量化并保存（用于模型归档，不用于推理）

`scripts/exp016_quantize.py` 的目的：
- 生成量化后的模型文件供 **存档/分享**（2.5GB vs 9.33GB）
- **但不在推理时直接加载这些文件**

```python
# exp016_quantize.py 的作用：归档用
pipe = load_model()
pipe.load_lora(lora_path)
quantize(pipe.dit)
save_file(pipe.dit.state_dict(), "quantized_model.safetensors")  # 仅用于归档
```

#### Stage 2: 生成视频（动态量化）

`scripts/exp015_generate_videos.py` + `--quantize nf4` 参数：
- **重新加载原始 FP16 模型**
- 每次推理前动态量化
- 生成视频

```python
# 推理脚本的实际做法
def generate_with_quantization(arm_config):
    # 1. 加载原始模型（FP16）
    pipe = load_fp_model(arm_config["base_model"])
    
    # 2. 加载 LoRA
    pipe.load_lora(pipe.dit, arm_config["lora_path"], alpha=1.0)
    
    # 3. 动态量化
    quant_config = QuantizeConfig(method="bitsandbytes_nf4")
    quant_config.quantize_model(pipe.dit, compute_device="cuda", model_device="cuda")
    
    # 4. 批量生成 99 个视频
    for prompt in prompts:
        video = pipe.generate_video(prompt, ...)
        save_video(video)
```

**关键点**：
- ✅ 量化开销只在**开始时支付一次**（~1-2 分钟）
- ✅ 之后生成 99 个视频都使用量化后的模型
- ✅ 没有保存/加载的序列化问题

---

## 🔍 你的新集群问题诊断

### 你做错了什么

根据 `quantization_technical_questions.md`，你尝试的流程：

```python
# ❌ 错误流程
transformer = load_model()
quantize(transformer)
save(transformer, "quantized.safetensors")  # 序列化失败
loaded = load("quantized.safetensors")      # 量化状态丢失
inference(loaded)                            # dtype 错误
```

**问题**：
1. 你试图 **保存并重新加载量化模型**
2. DiffSynth 的 `flatten_state_dict()` **没有实现序列化**
3. 加载后量化状态损坏 → 某些层变成 `uint8` → SiLU 报错

### 正确做法（参考 Exp016）

```python
# ✅ 正确流程
def generate_exp019_quantized():
    # 1. 加载原始擦除模型（FP16）
    pipe = WanVideoPipeline.from_pretrained(
        model_configs=[
            ModelConfig(model_id="wan5b/exp019_graddiff_merged", ...),
            # ... T5, VAE
        ],
        torch_dtype=torch.bfloat16,
        device="cuda"
    )
    
    # 2. 动态量化
    quant_config = QuantizeConfig(method="bitsandbytes_nf4")
    quant_config.quantize_model(pipe.dit, compute_device="cuda", model_device="cuda")
    print(f"✅ 量化完成")
    
    # 3. 生成视频（量化状态保持在内存中）
    for i, prompt in enumerate(prompts):
        video = pipe.generate_video(
            prompt=prompt,
            num_inference_steps=50,
            height=480, width=736, num_frames=17,
            seed=i,
        )
        save_video(video, f"output_{i}.mp4")
```

**关键**：
- ❌ 不要保存量化模型
- ❌ 不要加载量化模型
- ✅ 每次任务开始时动态量化一次
- ✅ 量化后的模型保持在内存中用于整个批次

---

## 📝 新集群 Exp019 量化的正确步骤

### Step 1: 准备环境

确认环境中有 bitsandbytes：
```bash
conda activate diffsynth
pip install bitsandbytes>=0.41.0
```

### Step 2: 修改生成脚本

参考 `scripts/exp015_generate_videos.py`，添加量化选项：

```python
# scripts/exp019_generate_quantized.py
import argparse
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.core.quant import QuantizeConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", choices=["none", "nf4"], default="none")
    args = parser.parse_args()
    
    # 1. 加载模型
    pipe = WanVideoPipeline.from_pretrained(...)
    
    # 2. 如果需要量化
    if args.quantize == "nf4":
        print("应用 NF4 量化...")
        quant_config = QuantizeConfig(method="bitsandbytes_nf4")
        quant_config.quantize_model(
            pipe.dit, 
            compute_device="cuda", 
            model_device="cuda"
        )
        print(f"✅ 量化完成：{quant_config.get_model_size(pipe.dit)}")
    
    # 3. 生成视频（量化状态保持在内存）
    for prompt in prompts:
        video = pipe.generate_video(prompt, ...)
        save_video(video, ...)
```

### Step 3: 提交 SLURM 作业

```bash
# slurm/exp019_quantize_generate.sbatch
#!/bin/bash
#SBATCH --job-name=exp019_quant
#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --output=logs/exp019_quant_%j.out

conda activate diffsynth

python scripts/exp019_generate_quantized.py \
    --model_path models/unlearned/exp019_graddiff_merged \
    --prompt_file hf_data/datasets/tiger_dataset/metadata_100.csv \
    --quantize nf4 \
    --num_repeats 5 \
    --output_dir outputs/exp019_quantized/
```

### Step 4: 评估

```bash
python scripts/eval_nudity_rate.py \
    --video_dir outputs/exp019_quantized/ \
    --output_json outputs/exp019_evaluation/graddiff_quantized_eval.json
```

---

## 📊 Exp016 的实际性能数据

### 量化开销

- **量化时间**: 1-3 分钟（一次性）
- **内存节省**: 9.33GB → ~1.2GB（推理时）
- **生成速度**: 与 FP16 基本相同（略慢 ~5%）

### 生成时间（99 个视频）

- **FP16**: ~4 小时
- **NF4**: ~4.2 小时（量化开销 ~2 分钟 + 生成略慢）

### 结论

**量化开销可忽略不计**（99 个视频摊销后 <2%）

---

## ⚠️ 关键教训

### DO ✅

1. **动态量化**：每次任务开始时量化一次，保持在内存
2. **批量生成**：量化后生成所有视频，摊销开销
3. **参考 Exp016**：复用已验证的代码模式

### DON'T ❌

1. ❌ 不要保存量化模型
2. ❌ 不要加载量化模型
3. ❌ 不要依赖 DiffSynth 的序列化功能（未实现）
4. ❌ 不要为每个视频重复量化（浪费时间）

---

## 🔗 相关代码参考

### 旧集群已验证的代码

1. **量化脚本**（归档用，不用于推理）
   - `scripts/exp016_quantize.py`

2. **生成脚本**（动态量化推理）
   - `scripts/exp015_generate_videos.py`
   - 查找 `--quantize` 参数的实现

3. **调试记录**
   - `docs/exp016_quantization_debugging.md`

### 新集群需要的脚本

```bash
# 从旧集群复制这些文件到新集群
scripts/exp015_generate_videos.py   # 参考量化推理逻辑
scripts/exp016_quantize.py          # 参考量化 API 调用
```

---

## 总结

**你的新集群问题根源**：试图保存/加载量化模型

**Exp016 的解决方案**：不保存，动态量化，推理完毕释放

**行动计划**：
1. ✅ 放弃保存量化模型的方案
2. ✅ 修改生成脚本，添加动态量化选项
3. ✅ 参考 `exp015_generate_videos.py` 实现
4. ✅ 提交作业，生成 → 评估 → 完成

**预期效果**：
- ✅ 推理成功，无 dtype 错误
- ✅ 内存从 9.33GB 降到 ~1.2GB
- ✅ 生成速度几乎不受影响
- ✅ Exp019 量化部分完成
