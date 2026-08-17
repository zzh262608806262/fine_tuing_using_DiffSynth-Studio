"""Safety Classifier 训练入口 (REINS Appendix C.2).

用法 (单卡):
  python -m classify.training.train \
      --config classify/configs/safety_classifier.yaml \
      --data.train_annotation data/safesora/train.jsonl \
      --data.test_annotation  data/safesora/test.jsonl \
      --data.video_root /path/to/videos

DDP (torchrun):
  torchrun --nproc_per_node=2 -m classify.training.train \
      --config ... --optim.amp true --train.ddp true ...

sanity check (不训练, 只跑 shape/冻结/NaN 校验):
  python -m classify.training.train --config ... --train.sanity_check true ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

# 让 `python -m classify.training.train` 与直接运行都能 import
_THIS_DIR = Path(__file__).resolve()
for _p in [_THIS_DIR.parents[3], _THIS_DIR.parents[2]]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from classify.factory import (
    build_model, build_transform, build_dataset, build_dataloader,
    build_optimizer, build_scheduler,
)
from classify.training.losses import BCEWithLogitsLoss
from classify.evaluation.metrics import compute_multilabel_metrics
from classify.utils import (
    set_seed, get_logger, configure_file_logger, log_json,
    load_yaml, apply_cli_overrides, is_main_process, get_world_size, barrier,
    save_checkpoint, load_checkpoint,
)


# =====================================================================
# DDP
# =====================================================================
def setup_ddp() -> bool:
    if not torch.cuda.is_available():
        return False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


def cleanup_ddp() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# =====================================================================
# Evaluation loop (在 test split 上)
# =====================================================================
@torch.no_grad()
def evaluate_loop(model, loader, device, label_names, threshold=0.5,
                  compute_auroc=True, compute_auprc=True):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        if batch.get("empty"):
            continue
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["labels"]
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = model(frames)
        probs = torch.sigmoid(logits.float())
        all_probs.append(probs.cpu())
        all_labels.append(labels)
    if not all_probs:
        return {"accuracy": 0.0, "warning": "no valid samples in eval"}
    probs = torch.cat(all_probs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return compute_multilabel_metrics(
        probs, labels, label_names,
        threshold=threshold,
        compute_auroc=compute_auroc, compute_auprc=compute_auprc,
    )


# =====================================================================
# Sanity checks (require.md 第十节)
# =====================================================================
def run_sanity_checks(model, loader, device, label_names, logger) -> None:
    logger.info("==== Sanity checks ====")
    model.eval()
    # 1. SigLIP frozen
    model.assert_backbone_frozen()
    logger.info("[ok] SigLIP backbone requires_grad 全部 False")

    # 2. trainable params
    trainable = model.trainable_parameters()
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"[ok] trainable params = {n_train:,} / total = {n_total:,} "
                f"({100.0 * n_train / max(1, n_total):.4f}%)")

    # 3. shapes / NaN
    batch = next(iter(loader))
    if batch.get("empty"):
        logger.error("sanity check: 第一个 batch 全部为损坏样本, 无法校验 shape. 请检查视频路径/解码器.")
        return
    frames = batch["frames"].to(device)
    labels = batch["labels"]
    B, T, C, H, W = frames.shape
    logger.info(f"[shape] frames         : {tuple(frames.shape)}  (期望 [B, T, 3, H, W])")
    frame_feats = model.backbone(frames)
    logger.info(f"[shape] SigLIP feature : {tuple(frame_feats.shape)}  (期望 [B, T, 768])")
    assert frame_feats.shape == (B, T, model.backbone.hidden_dim), "SigLIP 输出 shape 不符"
    t_out = model.temporal(frame_feats)
    logger.info(f"[shape] Temporal out    : {tuple(t_out.shape)}  (期望 [B, T, 768])")
    pooled = t_out.mean(dim=1)
    logger.info(f"[shape] mean pool       : {tuple(pooled.shape)}  (期望 [B, 768])")
    logits = model.head(pooled)
    logger.info(f"[shape] classifier logits: {tuple(logits.shape)}  (期望 [B, {model.num_classes}])")
    logger.info(f"[shape] labels          : {tuple(labels.shape)}  (期望 [B, {model.num_classes}])")
    assert labels.shape == (B, model.num_classes)
    # multi-hot 校验: 每样本至少一个 1 (safe 或 unsafe)
    sums = labels.sum(dim=1)
    logger.info(f"[ok] labels multi-hot: min_sum={float(sums.min())}, max_sum={float(sums.max())}")
    # NaN/Inf
    assert torch.isfinite(frame_feats).all(), "frame_feats 含 NaN/Inf"
    assert torch.isfinite(logits).all(), "logits 含 NaN/Inf"
    logger.info("[ok] 无 NaN/Inf")

    # 4. 前向 + loss
    loss_fn = BCEWithLogitsLoss()
    full_logits = model(frames)
    loss = loss_fn(full_logits, labels.to(device))
    logger.info(f"[ok] forward+loss: loss={float(loss):.4f}, logits={tuple(full_logits.shape)}")
    # 5. backward 只更新 temporal+head
    loss.backward()
    updated = 0
    for name, p in model.temporal.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            updated += 1
    head_updated = model.head.weight.grad is not None and model.head.weight.grad.abs().sum() > 0
    logger.info(f"[ok] temporal params with grad: {updated}; head.weight has grad: {head_updated}")
    logger.info("==== Sanity checks passed ====")


# =====================================================================
# Train
# =====================================================================
def train(cfg: Dict) -> None:
    distributed = setup_ddp()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    logger = get_logger("train")
    if is_main_process():
        configure_file_logger(cfg["train"]["output_dir"], "train")
        logger.info(f"Config:\n{json.dumps(cfg, ensure_ascii=False, indent=2, default=str)}")
        logger.info(f"world_size={get_world_size()}, device={device}")

    set_seed(int(cfg.get("seed", 42)))

    # ---- model ----
    model, label_names = build_model(cfg)
    model = model.to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[int(os.environ["LOCAL_RANK"])] if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )
    transform = build_transform(model.module if distributed else model, cfg)

    # ---- data ----
    train_ann = cfg["data"].get("train_annotation")
    test_ann = cfg["data"].get("test_annotation")
    if not train_ann or not test_ann:
        raise ValueError("必须提供 data.train_annotation 与 data.test_annotation")
    train_ds = build_dataset(cfg, train_ann, transform, label_names)
    test_ds = build_dataset(cfg, test_ann, transform, label_names)
    train_loader = build_dataloader(train_ds, cfg, train=True, distributed=distributed)
    test_loader = build_dataloader(test_ds, cfg, train=False, distributed=distributed)

    steps_per_epoch = max(1, len(train_loader))
    optimizer = build_optimizer(model.module if distributed else model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["optim"].get("amp", True)) and torch.cuda.is_available())
    loss_fn = BCEWithLogitsLoss()

    # ---- resume ----
    # epoch 末 checkpoint (batch_step=None): 从下一个 epoch 开始
    # mid-epoch checkpoint (batch_step=k):   重放同 epoch 的确定性顺序, 跳过前 k 个 batch
    start_epoch = 0
    best_metric = -1.0
    resume_batch_step = 0
    if cfg["train"].get("resume"):
        meta = load_checkpoint(cfg["train"]["resume"], model, optimizer, scheduler, scaler,
                               map_location="cpu")
        best_metric = float(meta.get("best_metric", -1.0))
        batch_step = meta.get("batch_step")
        if batch_step:
            start_epoch = int(meta.get("epoch", 0))
            resume_batch_step = int(batch_step)
            if distributed:
                # DistributedSampler 不支持样本级跳过, DDP 下退化为 epoch 级 resume
                logger.warning("DDP 下不支持 step 级 resume, 从该 epoch 开头重训")
                resume_batch_step = 0
        else:
            start_epoch = int(meta.get("epoch", 0)) + 1
        logger.info(f"Resumed from {cfg['train']['resume']}: epoch={start_epoch}, "
                    f"batch_step={resume_batch_step}, best={best_metric}")

    # ---- sanity check ----
    if cfg["train"].get("sanity_check"):
        run_sanity_checks(model.module if distributed else model, train_loader, device, label_names, logger)
        cleanup_ddp()
        return

    # ---- train loop ----
    grad_clip = float(cfg["optim"].get("grad_clip", 1.0))
    threshold = float(cfg["head"].get("threshold", 0.5))
    selection_metric = cfg["train"].get("selection_metric", "accuracy")
    epochs = int(cfg["optim"]["epochs"])
    log_every = int(cfg["train"].get("log_every", 20))
    save_every_steps = int(cfg["train"].get("save_every_steps", 0))  # 0 = 关闭 mid-epoch 保存
    bs = int(cfg["data"]["per_device_batch_size"])
    seed = int(cfg.get("seed", 42))
    last_path = os.path.join(cfg["train"]["output_dir"], "last.pt")

    for epoch in range(start_epoch, epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        # step 级 resume: 只在续跑的第一个 epoch 跳过已训过的 batch
        skip_batches = resume_batch_step if epoch == start_epoch else 0
        if hasattr(train_loader.sampler, "set_skip"):
            train_loader.sampler.set_skip(skip_batches * bs)
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_step = 0
        for step, batch in enumerate(train_loader, start=skip_batches):
            if batch.get("empty"):
                continue
            frames = batch["frames"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(frames)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.detach())
            n_step += 1
            if is_main_process() and (step % log_every == 0):
                lr = optimizer.param_groups[0]["lr"]
                logger.info(f"epoch {epoch} step {step}/{steps_per_epoch} "
                            f"loss={float(loss):.4f} lr={lr:.2e}")
            # mid-epoch checkpoint (step 级断点续跑; 中断最多丢 save_every_steps 步)
            if (save_every_steps > 0 and not distributed and is_main_process()
                    and (step + 1) % save_every_steps == 0 and (step + 1) < steps_per_epoch):
                save_checkpoint(
                    last_path, model, optimizer, scheduler, epoch, best_metric,
                    cfg, label_names, seed, scaler, batch_step=step + 1,
                )
                logger.info(f"[mid-epoch ckpt] epoch {epoch} batch_step={step + 1} -> {last_path}")
        barrier()
        train_loss = running_loss / max(1, n_step)

        # ---- eval ----
        metrics = evaluate_loop(
            model.module if distributed else model, test_loader, device, label_names,
            threshold=threshold,
            compute_auroc=cfg["eval"].get("compute_auroc", True),
            compute_auprc=cfg["eval"].get("compute_auprc", True),
        )
        if distributed:
            # 广播 metric (取 rank0)
            metric_tensor = torch.tensor([float(metrics.get(selection_metric, 0.0))],
                                         device=device)
            torch.distributed.broadcast(metric_tensor, src=0)
        cur_metric = float(metrics.get(selection_metric, 0.0))
        if is_main_process():
            log_json({"epoch": epoch, "train_loss": train_loss,
                      "eval_time_s": round(time.time() - t0, 1), **{f"eval/{k}": v for k, v in metrics.items()
                                                                     if not isinstance(v, dict)}},
                     step=epoch, logger=logger)
            if cur_metric > best_metric:
                best_metric = cur_metric
                ckpt_path = os.path.join(cfg["train"]["output_dir"], "best.pt")
                save_checkpoint(
                    ckpt_path,
                    model.module if distributed else model,
                    optimizer, scheduler, epoch, best_metric,
                    cfg, label_names, int(cfg.get("seed", 42)), scaler,
                )
                logger.info(f"[best] epoch {epoch} {selection_metric}={cur_metric:.4f} -> {ckpt_path}")
            # 始终保存 last (便于 resume; batch_step=None 表示该 epoch 已完成)
            save_checkpoint(
                last_path, model.module if distributed else model,
                optimizer, scheduler, epoch, best_metric,
                cfg, label_names, seed, scaler, batch_step=None,
            )
        barrier()

    if is_main_process():
        logger.info(f"Training done. best {selection_metric} = {best_metric:.4f}")
    cleanup_ddp()


# =====================================================================
# CLI
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Safety Classifier training (REINS Appendix C.2)")
    p.add_argument("--config", type=str, default="classify/configs/safety_classifier.yaml")
    # 覆盖项 (--data.xxx value) 不注册进 argparse, 统一从 unknown 里取,
    # 避免 argparse 把值吞成 positional 导致 flag/值配对错乱
    args, unknown = p.parse_known_args()
    args.overrides = list(unknown)
    return args


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
