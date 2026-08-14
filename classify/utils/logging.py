"""结构化日志 (仅 main process 输出, 避免多卡重复)."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .seed import is_main_process


def get_logger(name: str = "safety_classifier") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def configure_file_logger(log_dir: str, name: str = "safety_classifier") -> Optional[logging.Logger]:
    """在 main process 上额外把日志写入文件."""
    if not is_main_process():
        return None
    logger = get_logger(name)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    return logger


def log_json(metrics: Dict[str, Any], step: Optional[int] = None,
             logger: Optional[logging.Logger] = None) -> None:
    if not is_main_process():
        return
    logger = logger or get_logger()
    msg = json.dumps(metrics, ensure_ascii=False, default=str)
    prefix = f"[step {step}] " if step is not None else ""
    logger.info(prefix + msg)
