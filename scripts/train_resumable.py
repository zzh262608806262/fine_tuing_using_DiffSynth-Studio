"""
可断点续跑的 Wan 视频训练入口（不修改 DiffSynth-Studio 原训练代码）

复用 examples/wanvideo/model_training/train.py 的模型与数据集构建逻辑（通过 import，
不改动原文件），但训练循环替换为支持完整断点续跑的版本：

  - 通过 accelerator.save_state / load_state 保存并恢复
    模型权重 + 优化器动量 + LR scheduler + RNG 状态（兼容单卡 / DDP / DeepSpeed ZeRO）
  - progress.json 记录 epoch / epoch 内步数 / 全局步数，恢复后用
    accelerator.skip_first_batches 跳过当前 epoch 已训过的 batch
  - 数据顺序由按 epoch 设种子的 sampler 决定，恢复后 shuffle 顺序可复现
  - 状态目录原子覆盖（先写 .tmp 再 rename），中途被 kill 不会留下损坏的断点；
    只保留最新一份状态（注意：保存瞬间需要约 2 倍状态大小的磁盘空间）

用法（CLI 与原 train.py 完全一致，额外增加 3 个参数）:
    accelerate launch scripts/train_resumable.py \
        ... 原 train.py 的全部参数 ... \
        --resume auto --state_save_steps 200 --seed 42

    --resume auto   (默认) 若 output_path/resume_state 存在则自动恢复，否则从头训练
    --resume never  忽略已有断点，从头训练
    --state_save_steps N  每 N 步保存一次续跑状态（默认取 --save_steps；两者都为
                          None 时只在每个 epoch 结束保存）

限制:
  - 恢复时 GPU 数必须与保存时一致（DeepSpeed/DDP 分片状态与 world size 绑定）
  - 不支持 --enable_model_cpu_offload 和 data_process 任务（请用原 train.py）
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import accelerate
from tqdm import tqdm

WAN_TRAIN_PATH = os.path.join(ROOT, "examples", "wanvideo", "model_training", "train.py")
RESUME_STATE_DIRNAME = "resume_state"


def load_wan_train_module():
    """以模块形式加载原 train.py（不执行其 __main__ 部分），复用其组件。"""
    spec = importlib.util.spec_from_file_location("wan_train_original", WAN_TRAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SeededShuffleSampler(torch.utils.data.Sampler):
    """按 epoch 设种子的随机采样器：同一 epoch 的 shuffle 顺序跨进程、跨重启可复现，
    使 skip_first_batches 恢复后能准确跳到未训过的数据。"""

    def __init__(self, data_source_len, seed):
        self.data_source_len = data_source_len
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed * 100003 + self.epoch)
        yield from torch.randperm(self.data_source_len, generator=generator).tolist()

    def __len__(self):
        return self.data_source_len


def load_progress(state_dir):
    progress_path = os.path.join(state_dir, "progress.json")
    if not os.path.exists(progress_path):
        return None
    with open(progress_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_resume_state(accelerator, state_dir, progress):
    """保存完整训练状态。先写入 .tmp 目录，全部落盘后原子替换正式目录，
    保证任意时刻被 kill 都留有一份完整可用的断点。"""
    tmp_dir = state_dir + ".tmp"
    backup_dir = state_dir + ".old"
    if accelerator.is_main_process:
        for d in (tmp_dir, backup_dir):
            if os.path.exists(d):
                shutil.rmtree(d)
    accelerator.wait_for_everyone()
    accelerator.save_state(tmp_dir, safe_serialization=False)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        with open(os.path.join(tmp_dir, "progress.json"), "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
        if os.path.exists(state_dir):
            os.rename(state_dir, backup_dir)
        os.rename(tmp_dir, state_dir)
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        print(f"[resume] 断点已保存: {state_dir} (epoch={progress['epoch']}, "
              f"step_in_epoch={progress['step_in_epoch']}, global_step={progress['global_step']})")
    accelerator.wait_for_everyone()


def launch_resumable_training_task(accelerator, dataset, model, model_logger, sampler, args):
    from diffsynth.diffusion.runner import (
        initialize_deepspeed_gradient_checkpointing, save_training_args,
    )

    if accelerator.is_main_process:
        save_training_args(args)

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, collate_fn=lambda x: x[0],
        num_workers=args.dataset_num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler)
    initialize_deepspeed_gradient_checkpointing(accelerator)

    state_dir = os.path.join(args.output_path, RESUME_STATE_DIRNAME)
    start_epoch, resume_step_in_epoch, global_step = 0, 0, 0

    progress = load_progress(state_dir) if args.resume == "auto" else None
    if progress is not None:
        if progress["num_processes"] != accelerator.num_processes:
            raise RuntimeError(
                f"断点保存时使用 {progress['num_processes']} 个进程，"
                f"当前为 {accelerator.num_processes} 个。恢复训练必须使用相同的 GPU 数。")
        accelerator.load_state(state_dir)
        start_epoch = progress["epoch"]
        resume_step_in_epoch = progress["step_in_epoch"]
        global_step = progress["global_step"]
        model_logger.num_steps = global_step
        accelerator.print(
            f"[resume] 从断点恢复: epoch={start_epoch}, step_in_epoch={resume_step_in_epoch}, "
            f"global_step={global_step}")
    elif args.resume == "auto":
        accelerator.print(f"[resume] 未找到断点 ({state_dir})，从头开始训练。")
    else:
        accelerator.print("[resume] --resume never，从头开始训练。")

    steps_per_epoch = len(dataloader)
    if resume_step_in_epoch >= steps_per_epoch:
        start_epoch += 1
        resume_step_in_epoch = 0
    if start_epoch >= args.num_epochs:
        accelerator.print(f"[resume] 断点显示训练已完成 ({args.num_epochs} epochs)，无需继续。")
        return

    state_save_steps = args.state_save_steps if args.state_save_steps is not None else args.save_steps

    for epoch_id in range(start_epoch, args.num_epochs):
        # accelerate 包装后的 DataLoader 在 __iter__ 时会用自身的 iteration 计数器覆盖
        # sampler 的 epoch（重启进程后从 0 重数），必须通过它的 set_epoch 显式对齐，
        # 否则恢复后的 shuffle 顺序与保存前不一致。
        sampler.set_epoch(epoch_id)
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch_id)
        step_in_epoch = 0
        epoch_loader = dataloader
        if epoch_id == start_epoch and resume_step_in_epoch > 0:
            epoch_loader = accelerator.skip_first_batches(dataloader, resume_step_in_epoch)
            if hasattr(epoch_loader, "set_epoch"):
                epoch_loader.set_epoch(epoch_id)
            step_in_epoch = resume_step_in_epoch
        progress_bar = tqdm(
            epoch_loader, initial=step_in_epoch, total=steps_per_epoch,
            desc=f"Epoch {epoch_id}", disable=not accelerator.is_local_main_process)
        for data in progress_bar:
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, args.save_steps, loss=loss)
            step_in_epoch += 1
            global_step += 1
            if state_save_steps is not None and global_step % state_save_steps == 0:
                save_resume_state(accelerator, state_dir, {
                    "epoch": epoch_id, "step_in_epoch": step_in_epoch,
                    "global_step": global_step, "num_processes": accelerator.num_processes,
                })
        if args.save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
        save_resume_state(accelerator, state_dir, {
            "epoch": epoch_id + 1, "step_in_epoch": 0,
            "global_step": global_step, "num_processes": accelerator.num_processes,
        })

    model_logger.on_training_end(accelerator, model, args.save_steps)


def build_parser(wan_train):
    parser = wan_train.wan_parser()
    parser.add_argument("--resume", choices=["auto", "never"], default="auto",
                        help="auto=检测到断点自动恢复; never=忽略断点从头训练")
    parser.add_argument("--state_save_steps", type=int, default=None,
                        help="每 N 步保存续跑状态（默认取 --save_steps；均为 None 时仅每 epoch 保存）")
    parser.add_argument("--seed", type=int, default=42, help="数据 shuffle 种子")
    return parser


def main():
    wan_train = load_wan_train_module()
    from diffsynth.core import UnifiedDataset
    from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
    from diffsynth.diffusion.logger import ModelLogger

    args = build_parser(wan_train).parse_args()
    if args.task.endswith("data_process"):
        raise ValueError("data_process 任务无训练状态，请使用原 examples/wanvideo/model_training/train.py。")
    if args.enable_model_cpu_offload:
        raise ValueError("本脚本暂不支持 --enable_model_cpu_offload，请使用原 train.py 或关闭该选项。")

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(
            find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        }
    )
    model = wan_train.WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    sampler = SeededShuffleSampler(len(dataset), seed=args.seed)
    launch_resumable_training_task(accelerator, dataset, model, model_logger, sampler, args)


if __name__ == "__main__":
    main()
