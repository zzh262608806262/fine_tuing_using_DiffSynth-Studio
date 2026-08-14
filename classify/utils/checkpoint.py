"""Checkpoint 保存/加载/resume.

checkpoint 内容 (论文要求):
  - model state_dict
  - optimizer state_dict
  - scheduler state_dict
  - epoch
  - best metric
  - config
  - label mapping
  - random seed
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .seed import is_main_process


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_metric: float,
    config: Dict[str, Any],
    label_names,
    seed: int,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> None:
    """仅 main process 保存. DDP 时 model 需 unwrap."""
    if not is_main_process():
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    ckpt = {
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
        "label_names": list(label_names),
        "seed": seed,
    }
    torch.save(ckpt, path)
    # 同名 sidecar json, 便于人工查看
    meta = {k: v for k, v in ckpt.items() if k not in ("model_state_dict", "optimizer_state_dict",
                                                        "scheduler_state_dict", "scaler_state_dict")}
    with open(str(path) + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(ckpt["model_state_dict"], strict=True)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


def load_meta(path: str) -> Dict[str, Any]:
    """只读 meta (不加载权重), 用于 inference 取 label_names/config."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "config": ckpt.get("config"),
        "label_names": ckpt.get("label_names"),
        "epoch": ckpt.get("epoch"),
        "best_metric": ckpt.get("best_metric"),
        "seed": ckpt.get("seed"),
    }
