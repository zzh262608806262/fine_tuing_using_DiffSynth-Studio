"""随机种子控制 (论文 seed=42)。"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """设置 python / numpy / torch 随机种子。

    Args:
        seed: 随机种子, 论文默认 42.
        deterministic: 是否启用 cudnn deterministic (复现优先).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rank() -> int:
    """获取 DDP rank (单机单 GPU 返回 0)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def parse_optional_int(val: Optional[str]) -> Optional[int]:
    if val is None or val == "":
        return None
    return int(val)
