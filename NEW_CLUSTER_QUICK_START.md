# 新集群快速部署指南 - Tiger Dataset

**最后更新**: 2026-10-03  
**视频数据**: 已上传 HuggingFace ✅  
**预计时间**: 10-20 分钟

---

## 📦 数据下载

### Step 1: 下载 Tiger Dataset 视频

```bash
# 在新集群
cd /path/to/fine_tuing_using_DiffSynth-Studio

# 下载视频压缩包（349 MB）
hf download littlepig404/wan5b-unlearning-artifacts \
    datasets/tiger_dataset/tiger_dataset_videos.tar.gz \
    --repo-type dataset \
    --local-dir ./

# 解压视频文件
mkdir -p data/tiger_dataset
tar xzf datasets/tiger_dataset/tiger_dataset_videos.tar.gz -C data/tiger_dataset/

# 验证
ls data/tiger_dataset/*.mp4 | wc -l
# 应该输出: 1003
```

### Step 2: CSV 元数据已包含在 HF 数据中

```bash
# CSV 文件已在之前下载的 HF 数据中
ls hf_data/datasets/tiger_dataset/*.csv

# 复制到项目目录
cp hf_data/datasets/tiger_dataset/*.csv data/tiger_dataset/

# 验证
wc -l data/tiger_dataset/metadata*.csv
# 应该看到 7 个 CSV 文件
```

---

## 🚀 完整部署流程

### 一键部署脚本

创建文件 `setup_new_cluster.sh`：

```bash
#!/bin/bash
# 新集群快速部署脚本

set -e

PROJECT_ROOT="$(pwd)"
echo "项目根目录: $PROJECT_ROOT"

# ============================================
# 1. 下载 HuggingFace 数据
# ============================================
echo ""
echo "================================================"
echo "Step 1: 下载 HuggingFace 数据"
echo "================================================"

if [ ! -d "hf_data" ]; then
    echo "下载 HF 数据 (~4 GB)..."
    hf download littlepig404/wan5b-unlearning-artifacts \
        --repo-type dataset \
        --local-dir ./hf_data/
    echo "✓ HF 数据下载完成"
else
    echo "✓ HF 数据已存在，跳过下载"
fi

# ============================================
# 2. 解压 Tiger Dataset 视频
# ============================================
echo ""
echo "================================================"
echo "Step 2: 解压 Tiger Dataset 视频"
echo "================================================"

mkdir -p data/tiger_dataset

if [ -f "hf_data/datasets/tiger_dataset/tiger_dataset_videos.tar.gz" ]; then
    echo "解压视频文件 (1,003 个视频)..."
    tar xzf hf_data/datasets/tiger_dataset/tiger_dataset_videos.tar.gz \
        -C data/tiger_dataset/ \
        --strip-components=2
    echo "✓ 视频解压完成"
else
    echo "❌ 视频压缩包不存在"
    exit 1
fi

# ============================================
# 3. 复制 CSV 元数据
# ============================================
echo ""
echo "================================================"
echo "Step 3: 复制 CSV 元数据"
echo "================================================"

if [ -d "hf_data/datasets/tiger_dataset" ]; then
    cp hf_data/datasets/tiger_dataset/*.csv data/tiger_dataset/
    echo "✓ CSV 文件复制完成"
else
    echo "❌ CSV 文件不存在"
    exit 1
fi

# ============================================
# 4. 验证数据完整性
# ============================================
echo ""
echo "================================================"
echo "Step 4: 验证数据完整性"
echo "================================================"

# 检查视频文件数
VIDEO_COUNT=$(ls data/tiger_dataset/*.mp4 2>/dev/null | wc -l)
if [ "$VIDEO_COUNT" -eq 1003 ]; then
    echo "✓ 视频文件: $VIDEO_COUNT / 1003"
else
    echo "❌ 视频文件数量不匹配: $VIDEO_COUNT / 1003"
    exit 1
fi

# 检查 CSV 文件
CSV_COUNT=$(ls data/tiger_dataset/*.csv 2>/dev/null | wc -l)
if [ "$CSV_COUNT" -eq 7 ]; then
    echo "✓ CSV 文件: $CSV_COUNT / 7"
else
    echo "❌ CSV 文件数量不匹配: $CSV_COUNT / 7"
    exit 1
fi

# 检查 CSV 格式
echo "检查 CSV 格式..."
for csv in data/tiger_dataset/metadata*.csv; do
    HEADER=$(head -1 "$csv")
    if [[ "$HEADER" == "video,prompt"* ]]; then
        echo "  ✓ $(basename $csv)"
    else
        echo "  ❌ $(basename $csv) - 格式错误"
        exit 1
    fi
done

# ============================================
# 5. 测试数据加载
# ============================================
echo ""
echo "================================================"
echo "Step 5: 测试数据加载"
echo "================================================"

python3 << 'PYEOF'
import sys
import os
import pandas as pd

# 测试 CSV 加载
csv_file = 'data/tiger_dataset/metadata_100.csv'
try:
    df = pd.read_csv(csv_file)
    print(f"✓ CSV 加载成功: {len(df)} 条记录")
    print(f"  列名: {df.columns.tolist()}")
except Exception as e:
    print(f"❌ CSV 加载失败: {e}")
    sys.exit(1)

# 测试视频文件存在性
video_file = os.path.join('data/tiger_dataset', df['video'].iloc[0])
if os.path.isfile(video_file):
    file_size = os.path.getsize(video_file) / 1024
    print(f"✓ 视频文件可访问: {df['video'].iloc[0]} ({file_size:.1f} KB)")
else:
    print(f"❌ 视频文件不存在: {video_file}")
    sys.exit(1)

print("\n✓ 数据加载测试通过")
PYEOF

if [ $? -ne 0 ]; then
    echo "❌ 数据加载测试失败"
    exit 1
fi

# ============================================
# 完成
# ============================================
echo ""
echo "================================================"
echo "✓ 新集群部署完成！"
echo "================================================"
echo ""
echo "数据位置:"
echo "  - 视频: data/tiger_dataset/*.mp4 (1,003 个)"
echo "  - CSV: data/tiger_dataset/metadata*.csv (7 个)"
echo "  - HF 数据: hf_data/ (~4 GB)"
echo ""
echo "下一步:"
echo "  1. 配置环境: bash install_environment.sh"
echo "  2. 运行实验: 参考 memory/exp019_continuation_guide.md"
echo ""
```

保存并执行：

```bash
chmod +x setup_new_cluster.sh
./setup_new_cluster.sh
```

---

## 📋 手动部署步骤（如果脚本失败）

### Step 1: 克隆代码

```bash
cd ~
git clone https://github.com/zzh262608806262/fine_tuing_using_DiffSynth-Studio.git
cd fine_tuing_using_DiffSynth-Studio
```

### Step 2: 下载 HuggingFace 数据

```bash
# 安装 HF CLI
pip install huggingface_hub

# 下载完整数据 (~4 GB)
hf download littlepig404/wan5b-unlearning-artifacts \
    --repo-type dataset \
    --local-dir ./hf_data/

# 预计时间: 10-20 分钟
```

### Step 3: 解压视频文件

```bash
# 创建目录
mkdir -p data/tiger_dataset

# 解压视频 (1,003 个 MP4 文件)
tar xzf hf_data/datasets/tiger_dataset/tiger_dataset_videos.tar.gz

# 移动到正确位置
mv data/tiger_dataset/*.mp4 data/tiger_dataset/

# 或直接解压到目标位置
tar xzf hf_data/datasets/tiger_dataset/tiger_dataset_videos.tar.gz \
    -C data/tiger_dataset/ \
    --strip-components=2
```

### Step 4: 复制 CSV 文件

```bash
# CSV 已包含在 HF 数据中
cp hf_data/datasets/tiger_dataset/*.csv data/tiger_dataset/
```

### Step 5: 验证

```bash
# 检查视频数量
ls data/tiger_dataset/*.mp4 | wc -l
# 应该: 1003

# 检查 CSV
ls data/tiger_dataset/*.csv
# 应该看到:
# metadata.csv
# metadata_100.csv
# metadata_100_distill.csv
# metadata_30.csv
# metadata_3_smoketest.csv
# metadata_50_distill.csv
# metadata_70.csv

# 测试加载
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('data/tiger_dataset/metadata_100.csv')
print(f"✓ 加载成功: {len(df)} 条记录")
print(f"✓ 第一个视频: {df['video'].iloc[0]}")
EOF
```

---

## 🔧 环境配置

### Step 1: 安装依赖

```bash
# 自动安装脚本
bash install_environment.sh

# 或手动安装
conda env create -f environment.yml
conda activate diffsynth
pip install -r requirements.txt
```

### Step 2: 配置路径

```bash
# 复制环境变量模板
cp env.template .env

# 编辑 .env
vim .env
```

修改以下内容：

```bash
export PROJECT_ROOT="/path/to/fine_tuing_using_DiffSynth-Studio"
export VU_ROOT="${PROJECT_ROOT}/../video-unlearning"  # 如果有 VU 项目
export WAN_MODEL_PATH="${PROJECT_ROOT}/models/Wan-AI/Wan2.2-TI2V-5B"
export CONDA_ROOT="${HOME}/miniforge3"
export DIFFSYNTH_SKIP_DOWNLOAD="true"
```

加载环境变量：

```bash
source .env
```

### Step 3: 准备模型

```bash
# 从 HF 数据恢复模型
mkdir -p models/wan5b/

# 示例: Exp019 GradDiff 模型
cp -r hf_data/models/unlearned/exp019_graddiff/ \
      models/wan5b/exp019_graddiff_merged/

# 验证
ls models/wan5b/exp019_graddiff_merged/
# 应该看到:
# diffusion_pytorch_model-00001-of-00005.safetensors
# diffusion_pytorch_model-00002-of-00005.safetensors
# ...
# models_t5_umt5-xxl-enc-bf16.safetensors
# Wan2.2_VAE.safetensors
```

---

## 🧪 快速测试

### 测试 1: 数据加载

```bash
python3 << 'EOF'
import pandas as pd
import os

# 加载 CSV
df = pd.read_csv('data/tiger_dataset/metadata_100.csv')
print(f"✓ CSV: {len(df)} 条记录")

# 检查第一个视频
video_path = os.path.join('data/tiger_dataset', df['video'].iloc[0])
if os.path.exists(video_path):
    print(f"✓ 视频存在: {df['video'].iloc[0]}")
else:
    print(f"❌ 视频不存在: {video_path}")
EOF
```

### 测试 2: 视频读取

```bash
# 使用 ffprobe 检查视频
ffprobe data/tiger_dataset/BV1E3411u7tT_scene1_cut1.mp4

# 或用 Python
python3 << 'EOF'
import cv2
video_path = 'data/tiger_dataset/BV1E3411u7tT_scene1_cut1.mp4'
cap = cv2.VideoCapture(video_path)
if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✓ 视频可读: {frame_count} 帧, {fps} FPS")
    cap.release()
else:
    print("❌ 无法打开视频")
EOF
```

### 测试 3: 模型加载（可选）

```bash
# 测试 WanVideoPipeline 加载
python3 << 'EOF'
import os
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

model_configs = [
    ModelConfig(
        model_id="wan5b/exp019_graddiff_merged",
        origin_file_pattern="diffusion_pytorch_model*.safetensors"
    ),
    ModelConfig(
        model_id="wan5b/exp019_graddiff_merged",
        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.safetensors"
    ),
    ModelConfig(
        model_id="wan5b/exp019_graddiff_merged",
        origin_file_pattern="Wan2.2_VAE.safetensors"
    ),
]

tokenizer_config = ModelConfig(
    model_id="Wan-AI/Wan2.1-T2V-1.3B",
    origin_file_pattern="google/umt5-xxl/"
)

print("加载模型...")
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=model_configs,
    tokenizer_config=tokenizer_config,
)
print("✓ 模型加载成功")
EOF
```

---

## 📊 数据统计

部署完成后的目录结构：

```
fine_tuing_using_DiffSynth-Studio/
├── data/
│   └── tiger_dataset/
│       ├── *.mp4                          # 1,003 个视频 (643 MB)
│       ├── metadata.csv                   # 完整数据集
│       ├── metadata_100.csv               # 100 样本（评估）
│       ├── metadata_100_distill.csv       # 100 样本（蒸馏）
│       ├── metadata_70.csv                # 70 样本（擦除训练）
│       ├── metadata_50_distill.csv        # 50 样本（蒸馏）
│       ├── metadata_30.csv                # 30 样本（微调）
│       └── metadata_3_smoketest.csv       # 3 样本（测试）
├── hf_data/                               # HF 下载的数据 (~4 GB)
│   ├── datasets/
│   ├── models/
│   └── evaluations/
├── models/
│   └── wan5b/
│       └── exp019_graddiff_merged/        # 从 hf_data 复制
└── ...
```

**总占用空间**: ~5 GB
- HF 数据: 4 GB
- Tiger videos: 643 MB
- 其他: ~300 MB

---

## ⚠️ 常见问题

### Q1: tar 解压路径不对

如果解压后视频在 `data/tiger_dataset/data/tiger_dataset/*.mp4`：

```bash
# 移动到正确位置
mv data/tiger_dataset/data/tiger_dataset/*.mp4 data/tiger_dataset/
rm -rf data/tiger_dataset/data/
```

### Q2: HF 下载速度慢

使用镜像或分段下载：

```bash
# 方法 1: 使用镜像（如果有）
export HF_ENDPOINT=https://hf-mirror.com

# 方法 2: 只下载必需文件
hf download littlepig404/wan5b-unlearning-artifacts \
    --include "datasets/tiger_dataset/*" \
    --repo-type dataset \
    --local-dir ./hf_data/
```

### Q3: 视频文件损坏

验证完整性：

```bash
# 检查所有视频
for video in data/tiger_dataset/*.mp4; do
    if ! ffprobe "$video" 2>&1 | grep -q "Duration"; then
        echo "损坏: $video"
    fi
done
```

如果有损坏，重新下载：

```bash
rm data/tiger_dataset/*.mp4
tar xzf hf_data/datasets/tiger_dataset/tiger_dataset_videos.tar.gz -C data/tiger_dataset/
```

### Q4: 存储空间不足

最小化部署（仅 Exp019 需要）：

```bash
# 只解压 100 个评估视频
python3 << 'EOF'
import pandas as pd
import tarfile

df = pd.read_csv('hf_data/datasets/tiger_dataset/metadata_100.csv')
video_list = [f"data/tiger_dataset/{v}" for v in df['video']]

with tarfile.open('hf_data/datasets/tiger_dataset/tiger_dataset_videos.tar.gz') as tar:
    for member in tar.getmembers():
        if member.name in video_list:
            tar.extract(member, '.')
EOF
```

---

## ✅ 部署检查清单

- [ ] GitHub 代码已克隆
- [ ] HF 数据已下载 (~4 GB)
- [ ] Tiger 视频已解压 (1,003 个)
- [ ] CSV 文件已复制 (7 个)
- [ ] 视频数量验证通过 (1003)
- [ ] CSV 格式验证通过
- [ ] 数据加载测试通过
- [ ] 环境已配置 (.env)
- [ ] Conda 环境已创建
- [ ] 模型已从 HF 恢复

---

## 🚀 下一步

部署完成后，参考以下文档继续实验：

1. **Exp019 量化**: `memory/response_to_new_cluster_quantization_issues.md`
2. **Exp019 蒸馏**: `memory/exp019_continuation_guide.md`
3. **环境问题**: `NEW_CLUSTER_SETUP.md`
4. **数据说明**: `memory/tiger_dataset_migration_guide.md`

---

**HuggingFace 数据集**: https://huggingface.co/datasets/littlepig404/wan5b-unlearning-artifacts  
**GitHub 仓库**: https://github.com/zzh262608806262/fine_tuing_using_DiffSynth-Studio  
**预计部署时间**: 10-20 分钟  
**最后更新**: 2026-10-03
