# Exp016 量化调试记录

## 问题总结

Exp016 旨在对 Exp015 的 8 个模型臂进行 NF4 量化。经历了 7 次失败后，在 v8 中找到正确方案。

---

## v1: ModelScope 下载错误

**错误**:
```
[E3020] [404] 获取模型目录树失败：record not found
```

**原因**: 使用 `model_id="wan5b/exp015_esd_erased"`，diffsynth 尝试从 ModelScope 下载

**修复**: 改用 `ModelConfig(path=...)` 指向本地文件

---

## v2: 通配符未展开

**错误**:
```
FileNotFoundError: models/wan5b/exp015_esd_erased/diffusion_pytorch_model*.safetensors
```

**原因**: 通配符 `*` 直接传递给路径，未被 shell 展开

**修复**: 使用 `glob()` 展开通配符

---

## v3: 分片文件识别失败

**错误**:
```
ValueError: Cannot detect the model type
```

**原因**: 单个分片文件 `diffusion_pytorch_model-00001-of-00003.safetensors` 无法被 diffsynth 识别为完整模型

**修复**: 回退到 `model_id` 方式（参考 Exp015 的成功模式）

---

## v4: 仍触发下载

**错误**:
```
[E3020] [404] 获取模型目录树失败
```

**原因**: 即使使用 `model_id="wan5b/xxx"`，diffsynth 仍尝试从 ModelScope 下载

**修复**: 在 sbatch 中添加环境变量:
```bash
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
```

---

## v5: merge_lora 方法不存在

**Job ID**: 17451340

**错误**:
```
AttributeError: 'WanVideoPipeline' object has no attribute 'merge_lora'
```

**原因**: WanVideoPipeline 没有 `merge_lora()` 方法

**修复**: 移除 merge 步骤，直接加载 LoRA 后量化:
```python
pipe.load_lora(pipe.dit, lora_path, alpha=1.0)
quant_config.quantize_model(pipe.dit, ...)
```

---

## v6: quantize_model 导入错误

**Job ID**: 17451407

**错误**:
```
ImportError: cannot import name 'quantize_model' from 'diffsynth.core.quant'
```

**原因**: 量化不是独立函数，而是 `QuantizeConfig.quantize_model()` 方法

**检查 `quantize_wan5b.py` 发现正确用法**:
```python
quant_config = QuantizeConfig(method="bitsandbytes_nf4")
quant_config.quantize_model(pipe.dit, compute_device="cuda", model_device="cuda")
```

**修复**: 
- 移除错误的 `from diffsynth.core.quant import quantize_model`
- 改为调用 `quant_config.quantize_model(pipe.dit, ...)`

---

## v7: sbatch 模板变量未替换

**Job ID**: 17451448

**错误**:
```
exp016_quantize.py: error: argument --arm: invalid choice: 'ARMNAME'
```

**原因**: 直接 `sbatch slurm/exp016_quant_template.sbatch` 提交，模板中的 `ARMNAME` 没有被替换

**修复**: 使用 `sed` 替换变量:
```bash
sed 's/ARMNAME/esd_erased/g' slurm/exp016_quant_template.sbatch | sbatch
```

**创建批量提交脚本**: `scripts/submit_exp016_all.sh`

---

## v8: 正确方案 ✅

**Job ID**: 17451467 (esd_erased, 运行中)

**成功要素**:
1. **模型加载**: 使用 `model_id` + `DIFFSYNTH_SKIP_DOWNLOAD=true`
2. **LoRA 加载**: 只 `load_lora()`，不 merge
3. **量化 API**: `quant_config.quantize_model(pipe.dit, ...)`
4. **变量替换**: 使用 `sed` 替换模板变量

**核心代码**:
```python
# 构建配置
model_configs = [
    ModelConfig(model_id=base_model_id, origin_file_pattern=pattern)
    for pattern in MODEL_FILES
]

# 加载 pipeline
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=model_configs,
    tokenizer_config=tokenizer_config,
)

# 加载 LoRA（不 merge）
pipe.load_lora(pipe.dit, lora_path, alpha=1.0)

# 量化
quant_config = QuantizeConfig(method="bitsandbytes_nf4")
quant_config.quantize_model(pipe.dit, compute_device="cuda", model_device="cuda")

# 保存
save_file(pipe.dit.state_dict(), output_dir / "diffusion_pytorch_model.safetensors")
```

**提交方式**:
```bash
sed 's/ARMNAME/esd_erased/g' slurm/exp016_quant_template.sbatch | sbatch
```

---

## 关键教训

1. **分片模型加载**: 必须使用 `model_id` 方式，diffsynth 自动处理分片
2. **环境变量**: 本地模型需要 `DIFFSYNTH_SKIP_DOWNLOAD=true`
3. **量化 API**: 是 `QuantizeConfig` 对象的方法，不是独立函数
4. **LoRA 处理**: WanVideoPipeline 只有 `load_lora()`，没有 `merge_lora()`
5. **模板变量**: sbatch 模板需要显式替换变量，不能直接提交

---

## 待完成

- ✅ v8: esd_erased (Job 17451467, 运行中)
- ⏳ 其余 7 个臂: 等 v8 成功后批量提交
