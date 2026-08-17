"""共享工厂: 根据 config 构建 model / dataset / dataloader / optimizer / scheduler.

供 train.py / evaluate.py / predict.py 复用, 避免重复实现.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from .models.safety_classifier import SafetyClassifier
from .datasets.video_dataset import (
    FrameSampler, VideoDataset, VideoTransform, safe_collate,
)
from .utils import load_label_mapping, get_world_size, is_main_process, get_logger
from .utils.logging import get_logger as _gl

logger = _gl("factory")


def build_model(cfg: Dict) -> Tuple[SafetyClassifier, List[str]]:
    """构建 SafetyClassifier. 返回 (model, label_names)."""
    head = cfg["head"]
    label_names, num_classes = load_label_mapping(cfg["data"]["label_mapping"])
    # config 里的 num_classes 优先 (允许覆盖 label mapping)
    num_classes = int(head.get("num_classes", num_classes))
    model = SafetyClassifier(
        backbone_cfg={
            "model_name": cfg["backbone"]["model_name"],
            "hidden_dim": cfg["backbone"]["hidden_dim"],
            "image_size": cfg["backbone"]["image_size"],
            "feature_source": cfg["backbone"].get("feature_source", "patch_mean"),
            "freeze": cfg["backbone"].get("freeze", True),
            "dtype": cfg["backbone"].get("dtype", "float32"),
        },
        temporal_cfg=cfg["temporal"],
        num_classes=num_classes,
    )
    return model, label_names


def build_sampler(cfg: Dict) -> FrameSampler:
    v = cfg["video"]
    return FrameSampler(
        num_frames=v["num_frames"],
        sampling=v.get("sampling", "uniform"),
        short_clip_strategy=v.get("short_clip_strategy", "loop"),
    )


def build_transform(model: SafetyClassifier, cfg: Dict) -> VideoTransform:
    """从 backbone 的 image_processor 取归一化统计量."""
    image_size = cfg["backbone"]["image_size"]
    try:
        proc = model.backbone.get_image_processor()
        return VideoTransform.from_image_processor(proc, image_size=image_size)
    except Exception as e:
        logger.warning(f"取 image_processor 失败 ({e}), 使用默认 mean/std=0.5")
        return VideoTransform(image_size=image_size)


def build_dataset(cfg: Dict, annotation_path: str, transform: VideoTransform,
                  label_names: List[str]) -> VideoDataset:
    data_cfg = cfg["data"]
    sampler = build_sampler(cfg)
    return VideoDataset(
        annotation_path=annotation_path,
        video_root=data_cfg.get("video_root"),
        num_classes=cfg["head"].get("num_classes", len(label_names)),
        sampler=sampler,
        transform=transform,
        decode_backend=cfg["video"].get("decode_backend", "av"),
        max_decode_attempts=cfg["video"].get("max_decode_attempts", 2),
        label_names=label_names,
    )


class DeterministicShuffleSampler(torch.utils.data.Sampler):
    """按 (seed, epoch) 确定性 shuffle 的 sampler, 支持跳过前 skip 个样本.

    用于 step 级断点续跑: resume 时用相同 seed+epoch 重放当轮的样本顺序,
    并跳过已训过的前 k*batch_size 个样本, 从中断的 batch 继续.
    """

    def __init__(self, data_len: int, seed: int) -> None:
        self.data_len = data_len
        self.seed = seed
        self.epoch = 0
        self.skip = 0  # 以样本数计

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_skip(self, num_samples: int) -> None:
        self.skip = int(num_samples)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 100003 + self.epoch)
        order = torch.randperm(self.data_len, generator=g).tolist()
        return iter(order[self.skip:])

    def __len__(self) -> int:
        return self.data_len - self.skip


def build_dataloader(dataset: VideoDataset, cfg: Dict, train: bool,
                     distributed: bool = False) -> DataLoader:
    data_cfg = cfg["data"]
    bs = data_cfg["per_device_batch_size"]
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=train, drop_last=train
        )
    elif train:
        # 非 DDP 训练: 确定性 shuffle, 支持 step 级 resume
        sampler = DeterministicShuffleSampler(len(dataset), int(cfg.get("seed", 42)))
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        sampler=sampler,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        collate_fn=safe_collate,
        drop_last=train,
    )


def build_optimizer(model: SafetyClassifier, cfg: Dict):
    """AdamW, 仅训练 temporal + head (backbone 冻结)."""
    optim_cfg = cfg["optim"]
    params = model.trainable_parameters()
    lr = float(optim_cfg["lr_per_gpu"])
    if optim_cfg.get("scale_lr_by_world_size", True):
        lr = lr * get_world_size()
    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
        weight_decay=float(optim_cfg.get("weight_decay", 1e-2)),
    )
    return optimizer


def build_scheduler(optimizer, cfg: Dict, steps_per_epoch: int):
    optim_cfg = cfg["optim"]
    epochs = int(optim_cfg["epochs"])
    warmup_epochs = int(optim_cfg.get("warmup_epochs", 1))
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    from .training.scheduler import build_warmup_cosine_scheduler
    return build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
