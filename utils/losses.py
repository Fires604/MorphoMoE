"""Loss functions used by the submitted MorphoMoE experiments."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class Dice loss, combined with cross entropy during training."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_classes: Optional[int] = None,
    ) -> torch.Tensor:
        _, channels, _, _ = logits.shape
        if num_classes is None:
            num_classes = channels

        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(
            targets.clamp(min=0), num_classes=num_classes
        ).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_oh, dims)
        cardinality = torch.sum(probs + targets_oh, dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return (1.0 - dice).mean()
