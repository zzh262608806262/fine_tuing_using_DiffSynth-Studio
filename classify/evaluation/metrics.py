"""Multi-label 评估指标.

实现:
  - Accuracy (exact-match / subset accuracy)
  - Precision / Recall / F1 (macro & micro)
  - Hamming accuracy
  - AUROC (per-class, macro-avg)
  - AUPRC (per-class, macro-avg)
  - per-class metrics

推理: probs = sigmoid(logits); 默认 threshold=0.5 (configurable).
特别关注 unsafe categories 的 recall.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_multilabel_metrics(
    probs: np.ndarray,        # [N, C] sigmoid 概率
    labels: np.ndarray,       # [N, C] multi-hot 0/1
    label_names: List[str],
    threshold: float = 0.5,
    compute_auroc: bool = True,
    compute_auprc: bool = True,
) -> Dict:
    probs = _to_numpy(probs).astype(np.float64)
    labels = _to_numpy(labels).astype(np.int64)
    N, C = labels.shape
    assert probs.shape == labels.shape
    preds = (probs >= threshold).astype(np.int64)

    # ---- exact match (subset) accuracy ----
    exact_match = int((preds == labels).all(axis=1).sum())
    accuracy = exact_match / max(1, N)

    # ---- TP/FP/FN per class ----
    tp = ((preds == 1) & (labels == 1)).sum(axis=0)
    fp = ((preds == 1) & (labels == 0)).sum(axis=0)
    fn = ((preds == 0) & (labels == 1)).sum(axis=0)
    tn = ((preds == 0) & (labels == 0)).sum(axis=0)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    # 支持 (有正样本的类才计入 macro)
    support = labels.sum(axis=0)
    macro_mask = support > 0
    macro_precision = float(precision[macro_mask].mean()) if macro_mask.any() else 0.0
    macro_recall = float(recall[macro_mask].mean()) if macro_mask.any() else 0.0
    macro_f1 = float(f1[macro_mask].mean()) if macro_mask.any() else 0.0

    # micro
    tp_sum = int(tp.sum())
    fp_sum = int(fp.sum())
    fn_sum = int(fn.sum())
    micro_precision = tp_sum / max(1, tp_sum + fp_sum)
    micro_recall = tp_sum / max(1, tp_sum + fn_sum)
    micro_f1 = 2 * micro_precision * micro_recall / max(1e-12, micro_precision + micro_recall)

    # hamming accuracy (per-label accuracy)
    hamming = float((preds == labels).mean())

    metrics: Dict = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "hamming_accuracy": float(hamming),
    }

    # AUROC / AUPRC
    if compute_auroc:
        aurocs = []
        for c in range(C):
            if labels[:, c].sum() == 0 or labels[:, c].sum() == N:
                continue  # 该类无正/无负样本, 跳过
            try:
                from sklearn.metrics import roc_auc_score
                aurocs.append(roc_auc_score(labels[:, c], probs[:, c]))
            except Exception:
                pass
        metrics["macro_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")
    if compute_auprc:
        auprcs = []
        for c in range(C):
            if labels[:, c].sum() == 0:
                continue
            try:
                from sklearn.metrics import average_precision_score
                auprcs.append(average_precision_score(labels[:, c], probs[:, c]))
            except Exception:
                pass
        metrics["macro_auprc"] = float(np.mean(auprcs)) if auprcs else float("nan")

    # per-class
    per_class = {}
    for c in range(C):
        name = label_names[c] if c < len(label_names) else str(c)
        per_class[name] = {
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "support": int(support[c]),
        }
        if compute_auroc and labels[:, c].sum() not in (0, N):
            try:
                from sklearn.metrics import roc_auc_score, average_precision_score
                per_class[name]["auroc"] = float(roc_auc_score(labels[:, c], probs[:, c]))
                per_class[name]["auprc"] = float(average_precision_score(labels[:, c], probs[:, c]))
            except Exception:
                pass
    metrics["per_class"] = per_class

    # unsafe categories recall 汇总 (index 0 = safe, 其余 unsafe)
    unsafe_recall = []
    for c in range(1, C):
        if support[c] > 0:
            unsafe_recall.append(float(recall[c]))
    metrics["unsafe_recall_mean"] = float(np.mean(unsafe_recall)) if unsafe_recall else 0.0
    return metrics
