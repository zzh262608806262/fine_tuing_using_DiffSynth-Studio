# Exp 017 交互式节点调试计划

**目的**: 在 GPU 节点上一次性解决所有环境问题并测试训练命令

---

## 📋 步骤清单

### 1. 释放配额并申请节点

```bash
# 取消一个非关键 PENDING 作业（建议取消最早的）
scancel 17421467  # 或其他非紧急作业

# 申请交互式 GPU 节点
init1g

# 等待分配，记录节点名
# 假设分配到 nodeXXX

# 进入节点
jobsh nodeXXX
```

### 2. 测试并修复环境

```bash
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 激活环境
conda activate diffsynth

# 测试当前环境
python scripts/test_env.py
# 预期: pandas GLIBCXX 错误

# 修复 pandas
pip install 'pandas<2.1,>=2.0'

# 再次测试
python scripts/test_env.py
# 预期: 全部通过
```

### 3. 测试蒸馏训练命令（最小配置）

```bash
# 使用测试脚本（已配置为小数据集、1 epoch）
export ARM=base
bash scripts/exp017_interactive_test.sh

# 或手动运行
python scripts/distill_wan5b.py \
    --mode train \
    --no_download_dataset \
    --dataset_base_path data/tiger_dataset \
    --dataset_metadata_path data/tiger_dataset/metadata_100.csv \
    --output_path models/train/exp017_base_distill_test \
    --dataset_repeat 5 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --height 480 \
    --width 736 \
    --num_frames 17 \
    --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.safetensors,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.safetensors" \
    --use_gradient_checkpointing \
    --accelerate_config examples/wanvideo/model_training/full/accelerate_config_single_gpu.yaml
```

**观察重点**:
- ✅ 模型加载成功
- ✅ 训练开始（看到 "Epoch 1/1" 或 "step" 输出）
- ✅ 运行几个 step 后 loss 输出正常

**如果成功**: Ctrl+C 停止测试，准备提交 sbatch

### 4. 测试擦除模型蒸馏

```bash
# 测试 grad_ascent 擦除模型
export ARM=grad_ascent
python scripts/distill_wan5b.py \
    --mode train \
    --no_download_dataset \
    --dataset_base_path data/tiger_dataset \
    --dataset_metadata_path data/tiger_dataset/metadata_100.csv \
    --output_path models/train/exp017_grad_ascent_distill_test \
    --dataset_repeat 5 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --height 480 \
    --width 736 \
    --num_frames 17 \
    --model_id_with_origin_paths "models/wan5b/exp015_grad_ascent_erased:diffusion_pytorch_model*.safetensors,models/wan5b/exp015_grad_ascent_erased:models_t5_umt5-xxl-enc-bf16.safetensors,models/wan5b/exp015_grad_ascent_erased:Wan2.2_VAE.safetensors" \
    --use_gradient_checkpointing \
    --accelerate_config examples/wanvideo/model_training/full/accelerate_config_single_gpu.yaml
```

### 5. 清理并退出

```bash
# Ctrl+C 停止测试训练
exit  # 退出节点

# 释放节点（回到登录节点后）
scancel <交互式作业ID>
```

### 6. 提交正式作业

```bash
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
bash slurm/exp017_submit_all.sh
```

---

## 🔍 可能遇到的问题及解决

### 问题1: 仍然有 ImportError
- 检查具体错误信息
- 可能需要安装其他缺失的包

### 问题2: CUDA out of memory
- 检查是否有其他进程占用 GPU
- 可能需要减小 batch size（虽然配置中应该已优化）

### 问题3: 模型加载失败
- 检查模型路径是否正确
- 确认文件存在且完整

---

## 📝 记录测试结果

测试完成后记录：
- ✅ 环境修复成功
- ✅ base 模型蒸馏测试通过
- ✅ 擦除模型蒸馏测试通过
- 训练速度: ~X sec/step
- GPU 内存使用: ~X GB

然后可以放心提交完整的 sbatch 作业。

---

**预计时长**: 30-60 分钟（包括环境修复和命令测试）
