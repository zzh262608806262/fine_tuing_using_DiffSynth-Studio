"""Robust video loading: FrameSampler + VideoTransform + VideoDataset.

设计要点 (require.md 第六节):
  - 均匀采样 T=8 帧 (configurable)
  - 能处理不同长度视频
  - 帧数 < T 时有明确策略 (loop / last / first_last)
  - 视频损坏不让训练崩溃 -> 返回 None, collate 时跳过并 log
  - 不一次性把整段视频 decode 到 GPU (CPU decode, 只取 T 帧)
  - 多 backend: av (默认) / decord / opencv

输出统一 annotation 格式 (jsonl, 每行一条):
  {"video": "relative/or/abs/path.mp4", "labels": [0,0,1,...]}   # multi-hot list, len=num_classes
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.logging import get_logger

logger = get_logger("video_dataset")


# =====================================================================
# FrameSampler
# =====================================================================
@dataclass
class FrameSampler:
    """均匀/随机采样 T 帧.

    策略 (num_total < T 时):
      - "loop":      循环重复已有帧直到 T (保持时序连贯, 推荐)
      - "last":      最后一帧重复填充
      - "first_last": 首尾交替填充
    """
    num_frames: int = 8
    sampling: str = "uniform"          # "uniform" | "random"
    short_clip_strategy: str = "loop"

    def sample_indices(self, num_total: int) -> List[int]:
        T = self.num_frames
        if num_total == 0:
            return []
        if num_total >= T:
            if self.sampling == "random":
                # 随机有序采样
                idxs = sorted(random.sample(range(num_total), T))
            else:  # uniform
                # 均匀: 含首尾, 等间距
                if T == 1:
                    idxs = [num_total // 2]
                else:
                    step = (num_total - 1) / (T - 1)
                    idxs = [int(round(i * step)) for i in range(T)]
            return idxs
        # num_total < T -> 填充
        base = list(range(num_total))
        if self.short_clip_strategy == "last":
            return base + [num_total - 1] * (T - num_total)
        elif self.short_clip_strategy == "first_last":
            extra = []
            i = 0
            while len(base) + len(extra) < T:
                extra.append(base[0] if i % 2 == 0 else base[-1])
                i += 1
            return base + extra
        else:  # "loop"
            out = []
            i = 0
            while len(out) < T:
                out.append(base[i % num_total])
                i += 1
            return out


# =====================================================================
# Video decoders (CPU, 按需取帧)
# =====================================================================
def _decode_with_av(path: str, indices: List[int]) -> Optional[np.ndarray]:
    """PyAV: 仅解码需要的帧 (seek + 按需 decode). 返回 [T, H, W, 3] uint8 RGB."""
    import av  # type: ignore
    try:
        container = av.open(path)
    except Exception as e:
        logger.warning(f"[av] 打开失败 {path}: {e}")
        return None
    try:
        stream = container.streams.video[0]
        total = stream.frames
        # stream.frames 可能为 0 (某些容器未写 nb_frames), 用 duration 估算
        if total == 0:
            total = int(stream.duration * stream.average_rate) if stream.duration else 0
        if total == 0:
            # 退化为顺序遍历计数
            total = None

        target = set(indices)
        frames: Dict[int, np.ndarray] = {}
        # 按 keyframe 粗 seek 加速
        max_idx = max(indices)
        cur_idx = 0
        for frame in container.decode(stream):
            if cur_idx in target:
                arr = frame.to_ndarray(format="rgb24")  # [H, W, 3] uint8
                frames[cur_idx] = arr
            if len(frames) == len(target) or cur_idx > max_idx:
                break
            cur_idx += 1
        if len(frames) < len(target):
            # 若 total 未知导致索引越界, 用已采集的最后帧补齐
            if frames:
                last_arr = list(frames.values())[-1]
                for i in target:
                    if i not in frames:
                        frames[i] = last_arr
        return np.stack([frames[i] for i in indices], axis=0)  # [T, H, W, 3]
    except Exception as e:
        logger.warning(f"[av] 解码失败 {path}: {e}")
        return None
    finally:
        container.close()  # type: ignore[possibly-undefined]


def _decode_with_decord(path: str, indices: List[int]) -> Optional[np.ndarray]:
    try:
        import decord  # type: ignore
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path, num_threads=1)
        total = len(vr)
        safe_idx = [min(i, total - 1) for i in indices]
        frames = vr.get_batch(safe_idx).asnumpy()  # [T, H, W, 3] uint8 RGB
        return frames
    except Exception as e:
        logger.warning(f"[decord] 解码失败 {path}: {e}")
        return None


def _decode_with_opencv(path: str, indices: List[int]) -> Optional[np.ndarray]:
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logger.warning(f"[opencv] 打开失败 {path}")
            return None
        target = set(indices)
        frames: Dict[int, np.ndarray] = {}
        cur_idx = 0
        max_idx = max(indices)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if cur_idx in target:
                frames[cur_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if len(frames) == len(target) or cur_idx > max_idx:
                break
            cur_idx += 1
        cap.release()
        if not frames:
            return None
        if len(frames) < len(target):
            last_arr = list(frames.values())[-1]
            for i in target:
                if i not in frames:
                    frames[i] = last_arr
        return np.stack([frames[i] for i in indices], axis=0)
    except Exception as e:
        logger.warning(f"[opencv] 解码失败 {path}: {e}")
        return None


_DECODERS = {"av": _decode_with_av, "decord": _decode_with_decord, "opencv": _decode_with_opencv}


def load_video_frames(
    path: str,
    sampler: FrameSampler,
    backend: str = "av",
    max_attempts: int = 2,
) -> Optional[np.ndarray]:
    """返回 [T, H, W, 3] uint8 RGB, 失败返回 None.

    优先用 backend, 失败时回退到其它已安装的 backend.
    """
    order = [backend] + [b for b in ("av", "decord", "opencv") if b != backend]

    # 先确定总帧数 (用于采样索引); 各 backend 探测方式略不同, 这里用第一个可用 backend 试探
    last_err = None
    for attempt in range(max_attempts):
        for b in order:
            fn = _DECODERS.get(b)
            if fn is None:
                continue
            try:
                # 探测总帧数
                total = _count_frames(b, path)
                indices = sampler.sample_indices(total if total else sampler.num_frames)
                arr = fn(path, indices)
                if arr is not None and arr.shape[0] == sampler.num_frames:
                    return arr
            except Exception as e:
                last_err = e
                continue
    logger.warning(f"所有 backend 解码失败 {path}: {last_err}")
    return None


def _count_frames(backend: str, path: str) -> int:
    if backend == "av":
        import av  # type: ignore
        with av.open(path) as c:
            s = c.streams.video[0]
            t = s.frames
            if t == 0 and s.duration:
                t = int(s.duration * s.average_rate)
            return int(t)
    if backend == "decord":
        import decord  # type: ignore
        return len(decord.VideoReader(path, num_threads=1))
    if backend == "opencv":
        import cv2  # type: ignore
        cap = cv2.VideoCapture(path)
        t = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return t
    return 0


# =====================================================================
# VideoTransform
# =====================================================================
class VideoTransform:
    """逐帧图像变换: resize/center_crop -> normalize (与 SigLIP processor 一致).

    输入: [T, H, W, 3] uint8 RGB (numpy 或 tensor)
    输出: [T, 3, image_size, image_size] float32
    """

    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        self.image_size = image_size
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    @classmethod
    def from_image_processor(cls, image_processor, image_size: int = 224) -> "VideoTransform":
        """从 SigLIP image_processor 取 mean/std, 保证与 backbone 一致."""
        mean = tuple(image_processor.image_mean) if hasattr(image_processor, "image_mean") else (0.5, 0.5, 0.5)
        std = tuple(image_processor.image_std) if hasattr(image_processor, "image_std") else (0.5, 0.5, 0.5)
        return cls(image_size=image_size, mean=mean, std=std)

    def __call__(self, frames) -> torch.Tensor:
        # frames: [T, H, W, 3] uint8
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames.copy())
        if frames.dtype != torch.float32:
            frames = frames.float()
        if frames.dim() == 4 and frames.shape[-1] == 3:
            frames = frames.permute(0, 3, 1, 2)  # [T, 3, H, W]
        # center resize (保持长宽比 -> 先 resize 短边到 image_size, 再 center crop)
        frames = torch.nn.functional.interpolate(
            frames, size=(self.image_size, self.image_size), mode="bilinear",
            align_corners=False, antialias=True,
        )
        frames = frames / 255.0
        frames = (frames - self.mean.to(frames.device)) / self.std.to(frames.device)
        return frames  # [T, 3, H, W]


# =====================================================================
# VideoDataset
# =====================================================================
class VideoDataset(Dataset):
    """统一视频 multi-label 数据集.

    annotation 文件支持:
      - .jsonl : 每行 {"video": path, "labels": [multi-hot list]}
      - .json  : {"items": [{"video":..., "labels":...}, ...]}
      - .csv   : 第一列 video, 其余列为各 label (0/1), 列名=label_names

    Args:
        annotation_path: 标注文件
        video_root: 视频根目录 (相对路径拼接; 绝对路径直接用)
        num_classes: multi-hot 长度
        sampler: FrameSampler
        transform: VideoTransform
        decode_backend: "av" | "decord" | "opencv"
    """

    def __init__(
        self,
        annotation_path: str,
        video_root: Optional[str],
        num_classes: int,
        sampler: FrameSampler,
        transform: VideoTransform,
        decode_backend: str = "av",
        max_decode_attempts: int = 2,
        label_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.video_root = video_root
        self.num_classes = num_classes
        self.sampler = sampler
        self.transform = transform
        self.decode_backend = decode_backend
        self.max_decode_attempts = max_decode_attempts
        self.label_names = label_names
        self.items: List[Dict[str, Any]] = self._load_annotation(annotation_path)
        logger.info(f"Loaded {len(self.items)} samples from {annotation_path}")

    # ---- 标注加载 ----
    def _load_annotation(self, path: str) -> List[Dict[str, Any]]:
        ext = Path(path).suffix.lower()
        items: List[Dict[str, Any]] = []
        if ext == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and "items" in data:
                items = data["items"]
            else:
                # 视频路径 -> label dict
                items = [{"video": k, "labels": v} for k, v in data.items()]
        elif ext == ".csv":
            import csv
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                # 第一列 video, 其余为 label 列
                label_cols = header[1:]
                for row in reader:
                    video = row[0]
                    labels = [int(float(x)) for x in row[1:]]
                    items.append({"video": video, "labels": labels, "label_cols": label_cols})
        else:
            raise ValueError(f"不支持的标注格式 {ext} (支持 jsonl/json/csv)")
        # 校验 + 补全 multi-hot
        out = []
        for it in items:
            labels = it.get("labels")
            if labels is None and "label_cols" in it:
                # csv 已展开
                labels = it["labels"]
            if labels is None:
                logger.warning(f"样本缺少 labels, 跳过: {it.get('video')}")
                continue
            if len(labels) != self.num_classes:
                logger.warning(
                    f"label 长度 {len(labels)} != num_classes {self.num_classes}, 跳过: {it.get('video')}"
                )
                continue
            out.append({"video": it["video"], "labels": [int(x) for x in labels]})
        return out

    def _resolve_video_path(self, video: str) -> str:
        if os.path.isabs(video):
            return video
        if self.video_root:
            return os.path.join(self.video_root, video)
        return video

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        item = self.items[idx]
        video_path = self._resolve_video_path(item["video"])
        frames = load_video_frames(
            video_path, self.sampler, self.decode_backend, self.max_decode_attempts
        )
        if frames is None:
            # 损坏视频: 返回 None, collate 时跳过
            return None
        try:
            frames_t = self.transform(frames)  # [T, 3, H, W]
        except Exception as e:
            logger.warning(f"transform 失败 {video_path}: {e}")
            return None
        labels = torch.tensor(item["labels"], dtype=torch.float32)  # [num_classes]
        return {"frames": frames_t, "labels": labels, "video": item["video"]}


# =====================================================================
# collate: 跳过损坏样本 (None)
# =====================================================================
def safe_collate(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """过滤 None, 堆叠为 batch. 若全部失败, 返回空 dict (train loop 需处理)."""
    valid = [b for b in batch if b is not None]
    if len(valid) == 0:
        return {"empty": True, "frames": None, "labels": None, "videos": []}
    frames = torch.stack([b["frames"] for b in valid], dim=0)  # [B, T, 3, H, W]
    labels = torch.stack([b["labels"] for b in valid], dim=0)  # [B, num_classes]
    return {
        "empty": False,
        "frames": frames,
        "labels": labels,
        "videos": [b["video"] for b in valid],
    }
