"""LR schedule: 1-epoch linear warmup + cosine decay over 10 epochs.

论文 Appendix C.2:
  first 1 epoch: linear warmup
  remaining epochs: cosine decay
"""
from __future__ import annotations

from typing import Optional

from torch.optim.lr_scheduler import LambdaLR
import math


def build_warmup_cosine_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """linear warmup -> cosine decay.

    warmup 阶段: lr = base_lr * (step / warmup_steps)
    cosine 阶段: lr = base_lr * (min_lr_ratio + (1-min_lr_ratio)*0.5*(1+cos(pi*progress)))
    """
    assert total_steps > 0 and warmup_steps >= 0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos_val

    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)
