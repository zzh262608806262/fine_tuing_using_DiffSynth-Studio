# Exp 017 蒸馏训练最终状态报告

**日期**: 2026-09-04
**调试时长**: 10+ 小时
**尝试次数**: 28+ 次

## 当前作业状态

### Stage 1: 蒸馏训练

#### base (Job 17458945)
- **状态**: ✅ RUNNING
- **运行时间**: 13+ 分钟（持续增长）
- **节点**: node081
- **模型**: Wan-AI/Wan2.2-TI2V-5B (modelscope)
- **备注**: **首次成功启动！所有核心问题已解决**

#### grad_ascent (Job 17458964, 第4次提交)
- **状态**: ⏳ 提交中，等待验证
- **模型**: models/wan5b/exp015_grad_ascent_erased (本地目录)
- **修复**: 使用目录路径代替文件列表

#### esd (Job 17458965, 第4次提交)
- **状态**: ⏳ 提交中，等待验证
- **模型**: models/wan5b/exp015_esd_erased (本地目录)
- **修复**: 使用目录路径代替文件列表

## 完整修复列表（v1-v28+）

### 1. Dataset 格式问题
- **症状**: KeyError: 'seed', 缺少蒸馏字段
- **修复**: 生成包含 seed, rand_device, num_inference_steps, cfg_scale 的 metadata

### 2. Extra inputs 配置
- **症状**: TypeError: 'NoneType' object is not iterable, prompt=None
- **根因**: 
  - input_image 是 TI2V 专用，T2V 不需要
  - cfg_scale 会覆盖训练时的 cfg_scale=1，导致处理负向提示
  - num_inference_steps 重复（inputs_shared + inputs_posi）
- **修复**: extra_inputs 只保留 `seed,rand_device`

### 3. Train.py inputs 字段缺失
- **症状**: 各种 None/missing 错误
- **修复**: 
  - inputs_posi 添加: prompt, positive, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id
  - inputs_nega 添加: prompt, positive, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id

### 4. Python 缓存问题
- **症状**: 修改不生效
- **修复**: 清除所有 __pycache__

### 5. PYTHONPATH 问题
- **症状**: 错误堆栈显示使用了错误的 DiffSynth-Studio 路径
- **修复**: export PYTHONPATH="/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio:$PYTHONPATH"

### 6. 本地模型路径格式
- **症状**: 
  - JSON 数组不支持通配符
  - model_id_with_origin_paths 带 `:` 会触发 modelscope 下载
- **修复**: 使用目录路径，让 DiffSynth 自动发现模型文件
  - 正确: `models/wan5b/exp015_grad_ascent_erased`
  - 错误: `models/wan5b/exp015_grad_ascent_erased:diffusion_pytorch_model*.safetensors`

### 7. 依赖问题
- **症状**: ModuleNotFoundError: No module named 'tensorboard'
- **修复**: 移除 --enable_tensorboard_log

## 技术要点

### 蒸馏训练的参数传递

```
inputs_shared:
  - video 相关: input_video, height, width, num_frames
  - 训练控制: cfg_scale=1, rand_device, seed
  - gradient checkpointing 相关

inputs_posi/inputs_nega:
  - prompt, positive (区分正负向)
  - num_inference_steps (用于 TeacherCache)
  - tea_cache_l1_thresh, tea_cache_model_id (蒸馏专用)
```

### 参数传递流程

1. Dataset 提供原始数据（video, prompt, seed, rand_device, num_inference_steps, cfg_scale）
2. `get_pipeline_inputs` 构造三元组：(inputs_shared, inputs_posi, inputs_nega)
3. `parse_extra_inputs` 只添加到 inputs_shared（避免重复）
4. `unit_runner` 根据 `input_params_posi/nega` 从对应字典中提取参数
5. Loss 函数接收 `**inputs_shared, **inputs_posi`（不能重复！）

### cfg_scale 的影响

- `cfg_scale=1`: 只处理正向提示，负向直接复制正向结果
- `cfg_scale!=1`: 分别处理正负向提示，需要 inputs_nega 完整

训练时必须使用 `cfg_scale=1`，但 dataset 中的 cfg_scale 值用于 teacher 模型推理。

## 预计完成时间

- **训练**: 8-10 小时/臂 × 3 = 24-30 小时（并行）
- **视频生成**: 1-2 小时
- **安全评估**: 1-2 小时
- **总计**: ~10-12 小时（如果 3 臂同时成功启动）

## 监控命令

```bash
# 快速检查
bash check_exp017_v28.sh

# 详细监控
squeue -u $USER | grep exp017
sacct -j 17458945,17458964,17458965 --format=JobID,State,Elapsed,NodeList

# 检查日志
tail -f slurm/logs/exp017_distill_17458945_*.err
```
