"""
train.py

Importable training module for maskgen. Refactored from the original
argparse-based src/train.py to support notebook and Colab workflows.

Usage:
    from maskgen.train import train

    config = {
        "channels": 128,
        "layers": [1, 2, 4, 2],
        "stochastic_depth": 0.1,
        "ema": 0.9999,
        "img_size": 512,
        "crop_per_img": 4,
        "batch_size": 4,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "warmup_epoch": 5,
        "scheduler_params": {"T-max": 100},
        "epochs": 100,
        "gradient_clip": 1.0,
    }
    train(config, checkpoint_dir="./checkpoints", data_root="./data")
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import torch
import torch.amp as amp
import torch.optim as optim
import torch.optim.swa_utils as swa
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import ImageData
from .model import Net
from .utils import load_model, param_groups, save_model, set_device


def _detect_num_workers() -> int:
    """Auto-detect reasonable num_workers: 2 on Colab, up to 8 locally."""
    try:
        if "COLAB_RELEASE_TAG" in os.environ:
            return 2
    except Exception:
        pass
    cpu_count = os.cpu_count() or 2
    return min(cpu_count, 8)


def _train_epoch(
    model: torch.nn.Module,
    ema_model: swa.AveragedModel,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler.LRScheduler,
    scaler: amp.grad_scaler.GradScaler,
    device: str,
    config: dict,
) -> float:
    """Train one epoch."""
    model.train()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="Train", ncols=5)
    train_loss = -1

    clip_grad = config.get("clip_grad", config.get("gradient_clip"))

    for i, (images, masks) in enumerate(dataloader):
        optimizer.zero_grad()
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True, memory_format=torch.channels_last)

        with torch.autocast(device):
            output = model(images)
            loss = criterion(output, masks)

        scaler.scale(loss).backward()

        if clip_grad is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        scaler.step(optimizer)
        scaler.update()
        ema_model.update_parameters(model)

        batch_bar.set_postfix(
            loss=f"{loss.item()}",
            lr=f"{float(optimizer.param_groups[0]['lr'])}",
        )

        batch_bar.update()  # fixed: was missing ()
        train_loss = loss.item()

    if scheduler is not None:
        scheduler.step()

    batch_bar.close()

    return train_loss


@torch.no_grad()
def _validate(
    model: swa.AveragedModel,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> float:
    """Compute validation loss."""
    model.eval()

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="Val", ncols=5)
    val_loss = -1

    for i, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)

        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs, masks)

        batch_bar.set_postfix(loss=f"{loss.item()}")

        val_loss = loss.item()
        batch_bar.update()

    batch_bar.close()

    return val_loss


def train(
    config: dict,
    checkpoint_dir: str | Path = "./checkpoints",
    data_root: str | Path = "./data",
    resume_from: str | Path | None = None,
    use_wandb: bool = True,
) -> None:
    """
    Train the maskgen model.

    Args:
        config: Training configuration dict. Required keys:
            channels, layers, stochastic_depth, ema, img_size, crop_per_img,
            batch_size, lr, weight_decay, warmup_epoch, scheduler_params,
            epochs, gradient_clip (or clip_grad).
        checkpoint_dir: Directory to save checkpoints.
        data_root: Root directory containing train/val/test splits.
        resume_from: Path to a checkpoint to resume training from.
        use_wandb: Whether to log to Weights & Biases.
    """
    checkpoint_dir = Path(checkpoint_dir)
    data_root = str(data_root)

    # Optional wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb

            wandb = _wandb
        except ImportError:
            print("wandb not installed, skipping W&B logging. Install with: pip install wandb")
            wandb = None

    # Optional torchinfo
    torchinfo_summary = None
    try:
        from torchinfo import summary as torchinfo_summary
    except ImportError:
        pass

    # Global settings
    device = set_device()
    num_channels = config["channels"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Dataset and data loader
    num_workers = _detect_num_workers()
    print(f"Using {num_workers} data loading workers")

    train_dataset = ImageData(data_root, "train", True, config)
    val_dataset = ImageData(data_root, "val", False, config)

    train_loader = DataLoader(
        train_dataset, config["batch_size"], shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, config["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # Define model
    model = Net(num_channels, config)
    model = model.to(device=device, memory_format=torch.channels_last)

    if torchinfo_summary is not None:
        model_summary = str(torchinfo_summary(model, (config["batch_size"], 3, config["img_size"], config["img_size"])))
        arch_file = checkpoint_dir / "model_arch.txt"
        arch_file.write_text(model_summary)

    model = torch.compile(model)

    # Loss, optimizer, EMA, scaler, scheduler
    params = param_groups(model, config)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(params, lr=config["lr"])

    warmup = lr_scheduler.LinearLR(optimizer, total_iters=config["warmup_epoch"])
    cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["scheduler_params"]["T-max"])
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[config["warmup_epoch"]],
    )

    ema_model = swa.AveragedModel(model, multi_avg_fn=swa.get_ema_multi_avg_fn(config["ema"]))
    scaler = amp.grad_scaler.GradScaler(device)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = 1e9
    if resume_from is not None:
        model, ema_model, optimizer, scheduler, start_epoch, metrics = load_model(
            model, ema_model, device, optimizer=optimizer, scheduler=scheduler, path=str(resume_from)
        )
        best_val_loss = metrics.get("valid_loss", 1e9)
        start_epoch += 1  # start from the next epoch
        print(f"Resumed from {resume_from} at epoch {start_epoch}")

    # W&B init
    run = None
    if wandb is not None:
        wandb_params = config.get("wandb_params", {})
        wandb_key = config.get("wandb_key")
        if wandb_key:
            wandb.login(key=wandb_key)
        run = wandb.init(
            entity=wandb_params.get("entity"),
            name=wandb_params.get("name"),
            reinit=wandb_params.get("reinit", True),
            id=wandb_params.get("id"),
            resume=wandb_params.get("resume"),
            project=wandb_params.get("project", "maskgen"),
            config=config,
        )

    # Training loop
    metrics = {}
    for epoch in range(start_epoch, config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        train_loss = _train_epoch(
            model, ema_model, criterion, train_loader, optimizer, scheduler, scaler, device, config
        )
        curr_lr = float(optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch + 1}/{config['epochs']}: Train Loss {train_loss:.04f}\tLR {curr_lr:.04f}")

        valid_loss = _validate(ema_model, criterion, val_loader, device)

        metrics.update(
            {
                "lr": curr_lr,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "epoch": epoch,
            }
        )

        save_model(
            model, ema_model, optimizer, scheduler, metrics, epoch, str(checkpoint_dir / "last.pth")
        )
        print("Saved epoch model")

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            save_model(
                model, ema_model, optimizer, scheduler, metrics, epoch, str(checkpoint_dir / "best.pth")
            )
            print("Saved best val model")

        if run is not None:
            run.log(metrics)

    if run is not None:
        run.finish()
