我需要实现一个用于视频安全评估的 Safety Classifier，目标是复现论文：

“Pulling The REINS: Training-Free Safety Alignment of Video Diffusion Models via Representation Steering”

请严格按照论文 Appendix C.2 “Safety Classifier Training” 中描述的方案实现，不要自行改变核心架构。

我的最终用途是：
1. 对 Wan2.2 生成的视频进行安全分类；
2. 给 REINS 的 calibration set 自动生成 safety labels；
3. 后续用于 REINS Section 2.3 的 SPCA safety direction discovery；
4. 因此要求模型输出每个安全类别的独立概率，而不是只输出一个整体 safe/unsafe 分数。

========================
一、模型架构
========================

按照论文实现：

1. 输入：
   - 一个视频
   - 均匀采样 T=8 帧
   - T=8 必须 configurable，但默认值为 8

2. Vision Backbone：
   - 使用 pretrained SigLIP
   - hidden dimension d=768
   - SigLIP backbone 必须冻结
   - 不允许更新 SigLIP 参数
   - training 时保持 eval mode
   - 使用 torch.no_grad() 提取视觉特征
   - 请将 SigLIP 封装成独立模块，方便以后替换 checkpoint

3. Temporal modeling：
   - 在 SigLIP visual features 上使用 temporal Transformer encoder
   - Transformer layers = 4
   - attention heads = 8
   - activation = GELU
   - dropout = 0.1
   - 加入 sinusoidal temporal positional embeddings
   - Transformer 只负责建模 8 帧之间的时间关系

4. Temporal pooling：
   - Transformer 输出后沿 temporal dimension 做 mean pooling
   - 得到一个 d=768 的 video-level representation

5. Classification head：
   - 使用 Linear(768, num_classes)
   - 输出 per-category logits
   - 使用 sigmoid 得到每个类别独立概率
   - 必须是 multi-label classification，而不是 softmax single-label classification

========================
二、类别设置
========================

代码必须支持 configurable num_classes。

默认实现 SafeSora：

num_classes = 13

即：
safe + 12 unsafe categories

同时允许以后切换到 SafeWatch-Bench：

num_classes = 7

即：
safe + 6 unsafe categories

不要把类别名称硬编码在模型内部。
请单独建立 label mapping，例如：

label_names = [
    "safe",
    ...
]

具体类别名称从数据集 annotation/config 中读取。

========================
三、Loss
========================

使用：

BCEWithLogitsLoss

不要在计算 loss 前手动 sigmoid。

即：

logits = model(video)
loss = BCEWithLogitsLoss(logits, labels)

原因：
这是 multi-label classification，每个类别独立进行二分类。

labels shape：

[B, num_classes]

logits shape：

[B, num_classes]

训练和 inference 时分别处理：
- training: raw logits -> BCEWithLogitsLoss
- inference: sigmoid(logits) -> probabilities

========================
四、训练配置
========================

严格按照论文 Appendix C.2：

Optimizer:
AdamW

betas:
beta1 = 0.9
beta2 = 0.999

learning rate:
1e-5 per GPU

如果使用 DDP：
learning rate 按 world size 线性缩放：

lr = 1e-5 * world_size

weight decay:
1e-2

gradient clipping:
max L2 norm = 1.0

epochs:
10

Learning rate schedule:
- first 1 epoch: linear warmup
- remaining epochs: cosine decay

Per-device batch size:
16 videos

因为每个视频采样 8 帧：
16 videos × 8 frames = 128 frames/device/step

Mixed precision:
使用 autocast mixed precision

Random seed:
42

Checkpoint selection:
根据 held-out test split 上的 best accuracy 保存最佳 checkpoint。

========================
五、数据集
========================

优先实现 SafeSora。

要求：
- 使用 SafeSora 官方 13-label annotations
- 使用官方 train/test split
- 不要重新随机划分数据集
- dataset class 必须能够读取：
  video path
  multi-hot labels

SafeWatch-Bench 作为第二种可选数据源：
- 原始每个 clip 有 C1-C6 multi-hot annotations
- 转换为统一的 multi-label format
- 增加 derived “safe” indicator
- 通过 config 切换 SafeSora / SafeWatch-Bench

========================
六、视频读取与采样
========================

实现 robust video loader。

要求：
1. 均匀采样 8 帧；
2. 能处理不同长度视频；
3. 视频不足 8 帧时必须有明确处理策略；
4. 视频损坏时不能让整个训练过程崩溃；
5. 提供清晰的 error logging；
6. 支持常见 mp4 格式；
7. 不要一次性把整个视频全部 decode 到 GPU；
8. 尽量减少 CPU/GPU memory 使用。

请把 video decoding 单独封装，例如：

VideoDataset
FrameSampler
VideoTransform

========================
七、SigLIP feature extraction 的重要要求
========================

这里必须特别谨慎。

论文文字同时出现：
- patch-level features
- resulting (T,d) sequence
- d=768

请不要自行猜测 SigLIP 输出应该取哪个 tensor。

请先检查当前使用的 SigLIP Hugging Face implementation：
- model output structure
- last_hidden_state shape
- pooler_output shape
- vision tower hidden dimension

然后明确说明：
“论文中的 (T,d) sequence 在代码中对应 SigLIP 的哪个输出”。

如果原始 SigLIP 输出是：
[B, T, num_patches, D]

必须明确实现 patch aggregation 的方式，并在代码注释中解释。

不要静默地使用一个与论文不一致的 tensor。

========================
八、Temporal positional embedding
========================

实现 sinusoidal positional embedding：

shape:
[T, D]

并加入：

x = x + temporal_pos

要求：
- 不要使用可学习 positional embedding 替代；
- 默认 T=8；
- 代码支持修改 T。

========================
九、模型代码结构
========================

请按照清晰、模块化的方式组织代码：

project/
├── configs/
│   └── safety_classifier.yaml
├── datasets/
│   ├── safesora.py
│   ├── safewatch.py
│   └── video_dataset.py
├── models/
│   ├── siglip_backbone.py
│   ├── temporal_transformer.py
│   ├── positional_encoding.py
│   └── safety_classifier.py
├── training/
│   ├── train.py
│   ├── losses.py
│   └── scheduler.py
├── evaluation/
│   ├── evaluate.py
│   └── metrics.py
├── inference/
│   └── predict.py
├── utils/
│   ├── seed.py
│   ├── checkpoint.py
│   └── logging.py
└── README.md

如果你认为更好的目录结构，请说明原因后再调整。

========================
十、训练前必须实现 sanity checks
========================

在正式训练前实现：

1. dataset sample visualization
2. 检查 8 帧是否正确采样
3. 打印 SigLIP 输出 shape
4. 打印 Temporal Transformer 输入输出 shape
5. 打印 classifier logits shape
6. 检查 labels shape
7. 检查 labels 是否为 multi-hot
8. 检查是否存在 NaN/Inf
9. 检查 SigLIP requires_grad 是否全部为 False
10. 检查 temporal Transformer 和 classification head 是否正常更新

期望：

input:
[B, 8, ...]

SigLIP feature:
[B, 8, 768]

Temporal Transformer:
[B, 8, 768]

mean pooling:
[B, 768]

classifier:
[B, 13]

========================
十一、Evaluation
========================

不要只实现 accuracy。

至少实现：

- Accuracy
- Precision
- Recall
- F1
- Macro F1
- Micro F1
- AUROC（如果类别允许）
- AUPRC（如果类别允许）

同时输出：
- overall metrics
- per-class metrics

尤其关注 unsafe categories 的 recall。

推理时：
probabilities = sigmoid(logits)

默认 threshold = 0.5，
但 threshold 必须 configurable。

输出示例：

{
    "safe": 0.03,
    "violence": 0.91,
    "weapon": 0.82,
    ...
}

========================
十二、Checkpoint
========================

checkpoint 必须保存：

- model state_dict
- optimizer state_dict
- scheduler state_dict
- epoch
- best metric
- config
- label mapping
- random seed

提供：

train.py
evaluate.py
predict.py

三个独立入口。

========================
十三、Inference
========================

提供一个命令：

python inference/predict.py \
    --video path/to/video.mp4 \
    --checkpoint path/to/checkpoint.pt

输出：

1. 每个类别概率
2. predicted labels
3. overall safe/unsafe 判断
4. JSON 格式结果

例如：

{
    "video": "...",
    "predictions": {
        "safe": 0.01,
        "violence": 0.93,
        "weapon": 0.87
    },
    "unsafe": true
}

========================
十四、REINS compatibility
========================

这个 classifier 最终需要用于 REINS calibration。

因此提供一个 batch inference API：

输入：
一批生成的视频

输出：
每个视频的：
- multi-label probabilities
- binary safe/unsafe label

例如：

safe_label = 1 if predicted safe else 0

后续我会使用：

video -> safety classifier -> y

然后构建：

R ∈ R^(N×D)
Y ∈ R^(N×2)

用于 REINS Section 2.3 SPCA。

请确保 inference API 可以批量处理大量 Wan2.2 生成视频。

========================
十五、代码质量要求
========================

请不要只给我一个简化 demo。

我要的是：
“可以实际训练和评估”的工程代码。

要求：
- PyTorch
- Hugging Face Transformers
- 可运行
- 模块化
- 类型注解尽量完整
- 关键 tensor shape 必须写在注释中
- 关键数学操作解释清楚
- 不要隐藏异常
- 不要硬编码本地路径
- 所有路径通过 config / command line argument 指定
- 支持单 GPU
- 尽可能支持 DDP
- 支持 AMP mixed precision
- 支持 resume training

========================
十六、非常重要：先分析，再写代码
========================

不要直接开始生成大量代码。

请先完成以下工作：

Step 1:
分析论文 Appendix C.2 的 architecture，并列出最终 tensor shape。

Step 2:
检查 Hugging Face 当前 SigLIP API，确认：
- model name
- processor
- vision output
- hidden dimension
- patch token shape
- pooler output shape

Step 3:
指出论文文字中“patch-level features”和“(T,d)”之间可能存在的歧义，并说明你准备如何实现。

Step 4:
设计完整项目结构。

Step 5:
给出 config。

Step 6:
再逐文件实现代码。

每完成一个模块，都给出：
- 文件名
- 完整代码
- 输入 shape
- 输出 shape
- 为什么这样实现

如果论文描述和实际 Hugging Face API 存在冲突，不要擅自修改论文方案。
请明确指出冲突，并给出最接近论文的实现。