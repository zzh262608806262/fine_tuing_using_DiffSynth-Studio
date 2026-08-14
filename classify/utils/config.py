"""Config 加载: YAML + 命令行覆盖 + label mapping."""
from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError("PyYAML is required: pip install pyyaml") from e


def _interpolate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """简单变量插值: ${experiment_name} -> cfg["experiment_name"] (顶层)."""
    text = json.dumps(cfg, ensure_ascii=False)

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        # 仅支持顶层 key, 避免复杂度
        val = cfg.get(key, match.group(0))
        return str(val)

    text = re.sub(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}", _replace, text)
    return json.loads(text)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _interpolate(cfg)


def load_label_mapping(path: str) -> Tuple[List[str], int]:
    """读取 label mapping JSON. 返回 (label_names, num_classes).

    类别索引 = multi-hot 向量下标. 不要把类别名硬编码进模型.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    label_names: List[str] = data["label_names"]
    num_classes: int = int(data.get("num_classes", len(label_names)))
    assert len(label_names) == num_classes, (
        f"label_mapping 不一致: len(label_names)={len(label_names)} != num_classes={num_classes}"
    )
    return label_names, num_classes


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并 override 到 base (override 优先)."""
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def apply_cli_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """命令行覆盖: --data.per_device_batch_size 8 --head.num_classes 7.

    支持点号路径, 值自动推断 bool/int/float/str.
    """
    patch: Dict[str, Any] = {}
    i = 0
    while i < len(overrides):
        token = overrides[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:]
        if i + 1 < len(overrides) and not overrides[i + 1].startswith("--"):
            raw_val = overrides[i + 1]
            i += 2
        else:  # flag (bool)
            raw_val = "true"
            i += 1
        value = _parse_value(raw_val)
        # 写入嵌套 dict
        parts = key.split(".")
        cur = patch
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    return deep_update(cfg, patch)


def _parse_value(raw: str) -> Any:
    low = raw.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("none", "null"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw
