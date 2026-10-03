# Tiger Dataset 数据迁移指南

**创建日期**: 2026-09-28  
**目标**: 将 tiger_dataset 从旧集群迁移到新集群

---

## 📊 数据概览

### 当前位置（旧集群）

```
/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/data/tiger_dataset/
```

### 数据统计

- **视频文件**: 1,003 个 MP4 视频
- **总大小**: 643 MB
- **Metadata 文件**: 7 个 CSV

```bash
$ ls data/tiger_dataset/
├── *.mp4                              # 1,003 个视频文件
├── metadata.csv                       # 完整数据集 (1,003 条)
├── metadata_100.csv                   # 100 样本子集（生成评估用）
├── metadata_100_distill.csv           # 100 样本（蒸馏训练用）
├── metadata_70.csv                    # 70 样本（擦除训练用）
├── metadata_50_distill.csv            # 50 样本（蒸馏训练用）
├── metadata_30.csv                    # 30 样本（微调训练用）
└── metadata_3_smoketest.csv           # 3 样本（冒烟测试）
```

### 视频示例

```csv
video,prompt
BV1E3411u7tT_scene1_cut1.mp4,"The video depicts a group of people gathered around..."
BV15E411N7Hn_scene3_cut1.mp4,"The video shows a person wearing a white protective suit..."
```

---

## 🎯 数据用途

### 训练用途

| CSV 文件 | 用途 | 实验 | 视频数 |
|---------|------|------|--------|
| `metadata_70.csv` | 擦除训练 | Exp015 | 70 |
| `metadata_30.csv` | 微调训练 | Exp018 | 30 |
| `metadata_50_distill.csv` | 蒸馏训练（小） | Exp017 | 50 |
| `metadata_100_distill.csv` | 蒸馏训练（大） | Exp017 | 100 |
| `metadata_3_smoketest.csv` | 冒烟测试 | 所有 | 3 |

### 生成/评估用途

| CSV 文件 | 用途 | 实验 | 视频数 |
|---------|------|------|--------|
| `metadata_100.csv` | 视频生成 + 安全评估 | Exp015-019 | 100 |
| `metadata.csv` | 完整数据集（备用） | - | 1,003 |

---

## 📦 迁移方案

### 方案 1: 完整迁移（推荐）

**迁移所有 1,003 个视频 + 所有 CSV**

**优点**：
- ✅ 完整保留原始数据
- ✅ 支持所有实验配置
- ✅ 可以扩展到其他子集

**缺点**：
- ⚠️ 需要传输 643 MB

**步骤**：
```bash
# 在新集群
mkdir -p data/tiger_dataset/

# 方法 A: rsync（推荐，支持断点续传）
rsync -avP --progress \
    old_cluster:/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/data/tiger_dataset/ \
    data/tiger_dataset/

# 方法 B: scp
scp -r old_cluster:/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/data/tiger_dataset/ \
    data/

# 方法 C: tar 打包传输（最快）
# 旧集群：
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio
tar czf tiger_dataset.tar.gz data/tiger_dataset/
# 传输 tiger_dataset.tar.gz (约 500 MB 压缩后)
# 新集群：
tar xzf tiger_dataset.tar.gz
```

**验证**：
```bash
# 检查文件数
ls data/tiger_dataset/*.mp4 | wc -l
# 应该是: 1003

# 检查 CSV
wc -l data/tiger_dataset/metadata*.csv
# 应该看到 7 个文件，总计 1363 行
```

---

### 方案 2: 最小化迁移（仅必需文件）

**仅迁移实验需要的视频 + CSV**

适用于：
- 新集群存储受限
- 只需要完成特定实验

#### 选项 2A: Exp019 量化/蒸馏（最小集）

**需要的文件**：
- `metadata_100.csv` - 生成评估用（100 个视频）
- `metadata_50_distill.csv` - 蒸馏训练用（50 个视频）
- 对应的 150 个视频文件（去重后约 130 个）

**大小**: ~80 MB

**脚本**：
```bash
# 在旧集群生成文件列表
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

cat > /tmp/tiger_minimal_files.txt << 'EOF'
# CSV files
data/tiger_dataset/metadata_100.csv
data/tiger_dataset/metadata_50_distill.csv
data/tiger_dataset/metadata_3_smoketest.csv
EOF

# 从 CSV 提取视频文件名
awk -F, 'NR>1 {print "data/tiger_dataset/"$1}' \
    data/tiger_dataset/metadata_100.csv \
    data/tiger_dataset/metadata_50_distill.csv \
    data/tiger_dataset/metadata_3_smoketest.csv | \
    sort -u >> /tmp/tiger_minimal_files.txt

# 打包
tar czf tiger_minimal.tar.gz -T /tmp/tiger_minimal_files.txt

# 传输到新集群
# 新集群解压
tar xzf tiger_minimal.tar.gz
```

#### 选项 2B: 所有实验配置

**需要的文件**：
- 所有 7 个 CSV
- 对应的 ~200 个唯一视频（去重后）

**大小**: ~130 MB

**脚本**：
```bash
# 提取所有 CSV 中的视频
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

cat data/tiger_dataset/metadata*.csv | \
    awk -F, 'NR>1 && $1 ~ /\.mp4$/ {print "data/tiger_dataset/"$1}' | \
    sort -u > /tmp/tiger_videos_all.txt

# 添加 CSV 文件
find data/tiger_dataset -name "*.csv" >> /tmp/tiger_videos_all.txt

# 打包
tar czf tiger_all_configs.tar.gz -T /tmp/tiger_videos_all.txt
```

---

## 🚀 推荐迁移流程

### Step 1: 在旧集群打包

```bash
cd /home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio

# 完整打包（推荐）
tar czf tiger_dataset_full.tar.gz data/tiger_dataset/

# 查看大小
ls -lh tiger_dataset_full.tar.gz
# 预计: ~500 MB
```

### Step 2: 传输到新集群

```bash
# 方法 A: 直接 scp（如果网络好）
scp tiger_dataset_full.tar.gz new_cluster:/path/to/destination/

# 方法 B: 通过中转服务器
# 1. 上传到可访问的位置（如 HuggingFace）
hf upload littlepig404/wan5b-unlearning-artifacts \
    tiger_dataset_full.tar.gz \
    --repo-type dataset

# 2. 在新集群下载
hf download littlepig404/wan5b-unlearning-artifacts \
    tiger_dataset_full.tar.gz \
    --local-dir ./
```

### Step 3: 在新集群解压

```bash
cd /path/to/fine_tuing_using_DiffSynth-Studio

# 解压
tar xzf tiger_dataset_full.tar.gz

# 验证
ls data/tiger_dataset/*.mp4 | wc -l  # 应该是 1003
wc -l data/tiger_dataset/metadata*.csv  # 应该看到 7 个文件
```

### Step 4: 验证完整性

```bash
# 检查视频文件可读
for video in data/tiger_dataset/*.mp4; do
    if [ ! -r "$video" ]; then
        echo "❌ 无法读取: $video"
    fi
done

# 检查 CSV 格式
for csv in data/tiger_dataset/metadata*.csv; do
    echo "检查: $csv"
    head -1 "$csv"  # 应该是: video,prompt
done

# 测试加载
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('data/tiger_dataset/metadata_100.csv')
print(f"✓ 加载成功: {len(df)} 条记录")
print(f"✓ 列名: {df.columns.tolist()}")
EOF
```

---

## 📝 HuggingFace 集成（可选）

如果希望通过 HuggingFace 分发数据：

### 上传到 HuggingFace

```bash
# 准备上传目录
mkdir -p /tmp/hf_tiger_dataset
cp -r data/tiger_dataset/*.csv /tmp/hf_tiger_dataset/
cp -r data/tiger_dataset/*.mp4 /tmp/hf_tiger_dataset/

# 创建 README
cat > /tmp/hf_tiger_dataset/README.md << 'EOF'
# Tiger Dataset

1,003 video clips from various sources, used for video generation model training.

## Files

- `metadata.csv` - Full dataset (1,003 videos)
- `metadata_100.csv` - 100-sample subset for evaluation
- `metadata_70.csv` - 70-sample subset for unlearning training
- `metadata_30.csv` - 30-sample subset for fine-tuning
- `metadata_*_distill.csv` - Distillation training subsets
- `*.mp4` - Video files

## Usage

```python
import pandas as pd
df = pd.read_csv('metadata_100.csv')
for idx, row in df.iterrows():
    video_path = row['video']
    prompt = row['prompt']
    # Process video
```
EOF

# 上传（需要 git-lfs）
cd /tmp/hf_tiger_dataset
git lfs install
hf upload littlepig404/wan5b-unlearning-artifacts \
    . \
    --repo-type dataset \
    --path-in-repo datasets/tiger_dataset/
```

### 从 HuggingFace 下载

```bash
# 在新集群
hf download littlepig404/wan5b-unlearning-artifacts \
    --repo-type dataset \
    --include "datasets/tiger_dataset/*" \
    --local-dir ./hf_data/

# 复制到项目目录
mkdir -p data/tiger_dataset
cp -r hf_data/datasets/tiger_dataset/* data/tiger_dataset/
```

---

## ⚠️ 注意事项

### 1. 路径引用

所有脚本中的视频路径都是**相对路径**：
```python
# 脚本中的路径
video_path = "data/tiger_dataset/BV1E3411u7tT_scene1_cut1.mp4"

# 实际加载时
full_path = os.path.join(project_root, video_path)
```

**确保**：
- 项目根目录有 `data/tiger_dataset/` 目录
- CSV 中的视频文件名与实际文件匹配

### 2. 文件权限

```bash
# 确保文件可读
chmod 644 data/tiger_dataset/*.csv
chmod 644 data/tiger_dataset/*.mp4
```

### 3. 存储空间

检查新集群空间：
```bash
df -h /path/to/fine_tuing_using_DiffSynth-Studio
# 确保至少有 1 GB 可用空间
```

### 4. 视频完整性

如果传输后发现视频损坏：
```bash
# 检查视频可播放
for video in data/tiger_dataset/*.mp4; do
    ffprobe "$video" 2>&1 | grep -q "Duration" || echo "损坏: $video"
done
```

---

## 🔍 常见问题

### Q1: 为什么有 1,003 个视频但只用 100 个？

**A**: Tiger200K 是完整数据集，包含 1,003 个裁剪的视频片段。实验中选取不同子集用于不同目的：
- **100 个**：标准评估基准（`metadata_100.csv`）
- **70 个**：擦除训练（避免过拟合）
- **30 个**：微调训练（轻量级训练）
- **其他 903 个**：备用/扩展实验

### Q2: metadata_100.csv 和 metadata_100_distill.csv 有什么区别？

**A**: 两者包含相同的 100 个视频，但：
- `metadata_100.csv` - 用于**生成和评估**（仅 video, prompt 列）
- `metadata_100_distill.csv` - 用于**蒸馏训练**（额外包含 seed, rand_device, num_inference_steps, cfg_scale 列）

### Q3: 是否需要上传到 HuggingFace？

**A**: **不需要**，因为：
- 视频文件较大（643 MB）
- HuggingFace 已有必要的 CSV 元数据
- 视频仅用于训练，不需要公开分发

**推荐做法**：
- ✅ CSV 已上传 HF（724 KB）
- ❌ 视频文件不上传（在集群间直接传输）

### Q4: 如果只做 Exp019 量化，需要哪些视频？

**A**: 最少需要：
- `metadata_100.csv` + 对应 100 个视频（用于生成评估）
- 总计约 65 MB

**可选**：
- `metadata_50_distill.csv` + 对应 50 个视频（如果需要做蒸馏）

---

## 📋 迁移检查清单

- [ ] 旧集群打包完成
  ```bash
  tar czf tiger_dataset_full.tar.gz data/tiger_dataset/
  ls -lh tiger_dataset_full.tar.gz  # 应该 ~500 MB
  ```

- [ ] 传输到新集群
  ```bash
  scp tiger_dataset_full.tar.gz new_cluster:/path/
  ```

- [ ] 新集群解压
  ```bash
  tar xzf tiger_dataset_full.tar.gz
  ```

- [ ] 验证文件数
  ```bash
  ls data/tiger_dataset/*.mp4 | wc -l  # 1003
  ls data/tiger_dataset/*.csv | wc -l  # 7
  ```

- [ ] 验证完整性
  ```bash
  python3 -c "import pandas as pd; df = pd.read_csv('data/tiger_dataset/metadata_100.csv'); print(f'✓ {len(df)} rows')"
  ```

- [ ] 测试加载
  ```bash
  ffprobe data/tiger_dataset/BV1E3411u7tT_scene1_cut1.mp4
  # 应该显示视频信息
  ```

---

## 🎯 推荐流程总结

**最简单的方式**（新集群与旧集群网络互通）：

```bash
# 1. 在新集群直接 rsync
rsync -avP --progress \
    old_cluster:/home/x_jiage/jiage/fine_tuing_using_DiffSynth-Studio/data/tiger_dataset/ \
    data/tiger_dataset/

# 2. 验证
ls data/tiger_dataset/*.mp4 | wc -l  # 1003

# 3. 完成！
```

**预计时间**: 5-15 分钟（取决于网络速度）

---

**创建日期**: 2026-09-28  
**数据大小**: 643 MB（1,003 视频 + 7 CSV）  
**推荐方案**: 完整迁移（方案 1）
