"""Training entry point for the final MorphoMoE model only.

Run from the repository root, for example:
    python -m train.train --dataset neu --train-image-dir /path/train/images \
        --train-mask-dir /path/train/masks --val-image-dir /path/val/images \
        --val-mask-dir /path/val/masks
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Support both ``python -m train.train`` and ``python train/train.py``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.magnetic_tile_dataset import MagneticTileDataset
from datasets.neu_dataset import NEUSegDataset
from models.morphomoe import MorphoMoE
from utils.losses import DiceLoss
from utils.metrics import compute_confusion_matrix, metrics_from_confmat
from utils.utils import seed_dataloader_worker, set_random_seed


DATASET_NUM_CLASSES = {"neu": 4, "magnetic_tile": 6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the final MorphoMoE model")
    parser.add_argument("--dataset", choices=DATASET_NUM_CLASSES, default="neu")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--train-image-dir", type=Path, default=None)
    parser.add_argument("--train-mask-dir", type=Path, default=None)
    parser.add_argument("--val-image-dir", type=Path, default=None)
    parser.add_argument("--val-mask-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Initialize data/model/optimizer/loss and exit")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(config)
    for name, key in (("epochs", "epochs"), ("batch_size", "batch_size"),
                      ("lr", "learning_rate"), ("num_workers", "num_workers")):
        value = getattr(args, name)
        if value is not None:
            config[key] = value
    for argument, key in (
        ("train_image_dir", "train_image_dir"),
        ("train_mask_dir", "train_mask_dir"),
        ("val_image_dir", "val_image_dir"),
        ("val_mask_dir", "val_mask_dir"),
    ):
        value = getattr(args, argument)
        if value is not None:
            config[key] = str(value)
    if args.device is not None:
        config["device"] = args.device
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.run_name is not None:
        config["run_name"] = args.run_name
    config["dataset"] = args.dataset
    config["num_classes"] = DATASET_NUM_CLASSES[args.dataset]
    return config


def make_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    required_paths = ("train_image_dir", "train_mask_dir", "val_image_dir", "val_mask_dir")
    missing_paths = [name for name in required_paths if not config.get(name)]
    if missing_paths:
        raise ValueError(
            "Provide the following caller-defined directories in the YAML config "
            f"or command line: {', '.join(missing_paths)}"
        )
    dataset = config["dataset"]
    seed = int(config.get("seed", 42))

    if dataset == "neu":
        train_set = NEUSegDataset(config["train_image_dir"], config["train_mask_dir"])
        val_set = NEUSegDataset(config["val_image_dir"], config["val_mask_dir"])
    elif dataset == "magnetic_tile":
        train_set = MagneticTileDataset(config["train_image_dir"], config["train_mask_dir"])
        val_set = MagneticTileDataset(config["val_image_dir"], config["val_mask_dir"])
    else:  # argparse/config validation keeps this unreachable.
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed + 1)
    common = {"batch_size": int(config["batch_size"]), "num_workers": int(config["num_workers"]),
              "pin_memory": True, "worker_init_fn": seed_dataloader_worker}
    train_loader = DataLoader(train_set, shuffle=True, generator=train_generator, **common)
    val_loader = DataLoader(val_set, shuffle=False, generator=val_generator, **common)
    return train_loader, val_loader


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    optimizer = config.get("optimizer", "adam").lower()
    kwargs = {"lr": float(config["learning_rate"]), "weight_decay": float(config.get("weight_decay", 0.0))}
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError("optimizer must be 'adam' or 'adamw'")


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    scheduler = config.get("scheduler", "cosine").lower()
    if scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=float(config.get("lr_factor", 0.8)),
            patience=int(config.get("lr_patience", 8)), min_lr=float(config.get("min_lr", 1e-6)),
        )
    if scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(config["epochs"]), eta_min=float(config.get("min_lr", 1e-6)),
        )
    if scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, int(config["epochs"]) // 3), gamma=float(config.get("lr_factor", 0.8)),
        )
    if scheduler == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    raise ValueError("scheduler must be plateau, cosine, step, or exponential")


def resize_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != target.shape[-2:]:
        return F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def train_one_epoch(model, loader, optimizer, ce_loss, dice_loss, device, num_classes) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        logits, _ = model(images, return_aux=True)
        logits = resize_logits(logits, masks)
        loss = ce_loss(logits, masks) + dice_loss(logits, masks, num_classes=num_classes)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, ce_loss, dice_loss, device, num_classes):
    model.eval()
    total_loss = 0.0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    for batch in tqdm(loader, desc="validate", leave=False):
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        logits, _ = model(images, return_aux=True)
        logits = resize_logits(logits, masks)
        loss = ce_loss(logits, masks) + dice_loss(logits, masks, num_classes=num_classes)
        total_loss += loss.item() * images.size(0)
        confusion += compute_confusion_matrix(logits.argmax(dim=1), masks, num_classes).to(device)
    return total_loss / len(loader.dataset), metrics_from_confmat(confusion.cpu())


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics, config: Dict[str, Any]) -> None:
    torch.save({
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "model_class": "MorphoMoE", "model_config": model.get_model_config(),
        "epoch": epoch, "metrics": {key: float(value) for key, value in metrics.items() if value.ndim == 0},
        "config": config,
    }, path)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config or root / "configs" / f"{args.dataset}.yaml"
    config = apply_cli_overrides(load_config(config_path), args)
    set_random_seed(int(config.get("seed", 42)))
    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader = make_dataloaders(config)
    model = MorphoMoE(in_channels=3, num_classes=int(config["num_classes"]), bilinear=True).to(device)
    optimizer = create_optimizer(model, config)
    ce_loss, dice_loss = nn.CrossEntropyLoss(), DiceLoss()
    print(f"dataset={config['dataset']} device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    if args.dry_run:
        print(f"initialized train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
        return

    scheduler = create_scheduler(optimizer, config)
    output_dir = Path(config.get("output_dir", "outputs")) / config["dataset"]
    if config.get("run_name"):
        output_dir /= str(config["run_name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_miou, patience = -float("inf"), 0
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, ce_loss, dice_loss, device, config["num_classes"])
        val_loss, metrics = evaluate(model, val_loader, ce_loss, dice_loss, device, config["num_classes"])
        miou = float(metrics["mIoU"])
        if config.get("scheduler", "cosine").lower() == "plateau":
            scheduler.step(miou)
        else:
            scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               **{key: float(value) for key, value in metrics.items() if value.ndim == 0}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if miou > best_miou:
            best_miou, patience = miou, 0
            save_checkpoint(output_dir / "best.pth", model, optimizer, epoch, metrics, config)
        else:
            patience += 1
        max_patience = int(config.get("early_stopping_patience", 0))
        if max_patience > 0 and patience >= max_patience:
            break
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
