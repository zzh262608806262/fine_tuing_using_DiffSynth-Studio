"""Safety Classifier for video safety alignment (REINS Appendix C.2 reproduction).

子模块:
  - models:    SigLIP backbone (frozen) + temporal Transformer + linear head
  - datasets:  robust video loader + SafeSora / SafeWatch 适配
  - training:  AdamW + warmup-cosine, AMP, DDP, BCEWithLogitsLoss
  - evaluation: multi-label metrics (acc/P/R/F1/macro/micro/AUROC/AUPRC/per-class)
  - inference:  single & batch predict + REINS SPCA 兼容 API
"""
from . import models
from . import datasets
from . import utils

__all__ = ["models", "datasets", "utils"]
