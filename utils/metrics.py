"""Segmentation metrics used by the submitted MorphoMoE experiments."""

from typing import Dict

import torch


@torch.no_grad()
def compute_confusion_matrix(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Return a confusion matrix whose rows are targets and columns predictions."""
    pred = pred.view(-1)
    target = target.view(-1)
    valid = (target >= 0) & (target < num_classes)
    indices = num_classes * target[valid] + pred[valid]
    return torch.bincount(indices, minlength=num_classes ** 2).reshape(
        num_classes, num_classes
    )


@torch.no_grad()
def metrics_from_confmat(cm: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute mIoU, mDice, accuracy, and their per-class values."""
    true_positive = cm.diag()
    support = cm.sum(dim=1)
    predicted = cm.sum(dim=0)
    union = support + predicted - true_positive
    iou = true_positive.float() / union.clamp(min=1)
    dice = (2 * true_positive.float()) / (support + predicted).clamp(min=1)
    class_accuracy = true_positive.float() / support.clamp(min=1)
    return {
        "mIoU": iou.mean(),
        "mDice": dice.mean(),
        "Acc": true_positive.sum().float() / cm.sum().clamp(min=1),
        "IoU_per_class": iou,
        "Dice_per_class": dice,
        "Acc_per_class": class_accuracy,
    }
