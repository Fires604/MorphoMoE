"""Standalone final MoESteelUNet matching the visible topology of paper Figure 5.

The model contains a shared multi-scale encoder, three heterogeneous experts
(SDE, RDE and CSE), sample-level soft routing, and a U-Net decoder.  Its CSE is
the executable interpretation that produced the stronger experiment result:

    Deformable Conv3x3 -> fixed Sobel-X/Y -> Conv3x3 + BN + ReLU
    -> Conv1x1 + ReLU + Conv1x1 + Sigmoid shape-context gate.

Figure 5 does not identify the deformable-convolution revision or the exact
X/Y fusion and gate application.  This implementation records the choices
explicitly: unmodulated DCNv1-style offsets, fixed depthwise Sobel filters,
direction-preserving concatenation, and multiplicative shape gating.

This file is self-contained so the model can be copied to another project
without importing another MoESteelUNet implementation.  Forward returns raw
multiclass logits, as expected by ``CrossEntropyLoss``.
"""

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import DeformConv2d
except (ImportError, RuntimeError) as exc:  # pragma: no cover - environment dependent
    DeformConv2d = None
    _DEFORM_CONV_IMPORT_ERROR = exc
else:
    _DEFORM_CONV_IMPORT_ERROR = None


EXPERT_ORDER: Tuple[str, str, str] = ("sde", "rde", "cse")


def _conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int = 0,
    dilation: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            _conv_bn_relu(in_channels, out_channels, 3, padding=1),
            _conv_bn_relu(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleEncoderBlock(nn.Module):
    """1x1 projection followed by channel-split 3/5/7/9 convolution paths."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by four")
        quarter = out_channels // 4
        self.pre_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.branches = nn.ModuleList(
            [
                _conv_bn_relu(quarter, quarter, kernel, padding=kernel // 2)
                for kernel in (3, 5, 7, 9)
            ]
        )
        self.fuse_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.pre_conv(x)
        chunks = torch.chunk(local, chunks=4, dim=1)
        processed = [branch(chunk) for branch, chunk in zip(self.branches, chunks)]
        return self.fuse_conv(torch.cat(processed, dim=1))


class SharedEncoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.stage1 = MultiScaleEncoderBlock(in_channels, 16)
        self.stage2 = nn.Sequential(nn.MaxPool2d(2), MultiScaleEncoderBlock(16, 32))
        self.stage3 = nn.Sequential(nn.MaxPool2d(2), MultiScaleEncoderBlock(32, 64))
        self.stage4 = nn.Sequential(nn.MaxPool2d(2), MultiScaleEncoderBlock(64, 128))

    def forward(self, x: torch.Tensor):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return x4, (x1, x2, x3, x4)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = self.mlp(F.adaptive_avg_pool2d(x, 1))
        maximum = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(average + maximum)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        descriptor = torch.cat(
            [x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1
        )
        return x * torch.sigmoid(self.conv(descriptor))


class CBAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class SmallDefectExpert(nn.Module):
    """Standard/dilated small-defect branches followed by CBAM."""

    def __init__(self, in_channels: int = 128, out_channels: int = 64):
        super().__init__()
        self.standard_branch = nn.Sequential(
            _conv_bn_relu(in_channels, out_channels, 3, padding=1),
            _conv_bn_relu(out_channels, out_channels, 3, padding=1),
        )
        self.dilated_branch = nn.Sequential(
            _conv_bn_relu(in_channels, out_channels, 3, padding=2, dilation=2),
            _conv_bn_relu(out_channels, out_channels, 1),
        )
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False)
        self.attention = CBAM(out_channels)

    def forward(self, deepest_feature: torch.Tensor) -> torch.Tensor:
        standard = self.standard_branch(deepest_feature)
        dilated = self.dilated_branch(deepest_feature)
        return self.attention(self.fusion(torch.cat([standard, dilated], dim=1)))


class SEConv3x3(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.feature = _conv_bn_relu(channels, channels, 3, padding=1)
        hidden = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature(x)
        return feature * self.se(feature)


class RegionalDefectExpert(nn.Module):
    """Repeated partial-channel large-kernel regional-defect path."""

    def __init__(self, in_channels: int = 128, out_channels: int = 64):
        super().__init__()
        if in_channels % 4 != 0:
            raise ValueError("in_channels must be divisible by four")
        quarter = in_channels // 4
        self.quarter_channels = quarter
        self.pre_conv = _conv_bn_relu(in_channels, in_channels, 1)
        self.large_branch = nn.Sequential(
            _conv_bn_relu(quarter, quarter, 13, padding=6),
            _conv_bn_relu(quarter, quarter, 1),
        )
        self.se_branch = SEConv3x3(quarter)
        self.fuse = _conv_bn_relu(quarter * 2, quarter, 1)
        self.post_conv = _conv_bn_relu(in_channels, out_channels, 1)

    def forward(self, deepest_feature: torch.Tensor) -> torch.Tensor:
        x = self.pre_conv(deepest_feature)
        for _ in range(2):
            processed_input = x[:, : self.quarter_channels]
            retained = x[:, self.quarter_channels :]
            large = self.large_branch(processed_input)
            local = self.se_branch(processed_input)
            processed = self.fuse(torch.cat([large, local], dim=1))
            x = torch.cat([processed, retained], dim=1)
        return self.post_conv(x)


class FixedSobelDepthwise(nn.Module):
    """Apply fixed Sobel-X/Y kernels independently to every feature channel."""

    def __init__(self, channels: int):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        )
        sobel_y = sobel_x.t().contiguous()
        self.channels = channels
        self.register_buffer(
            "sobel_x", sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        )
        self.register_buffer(
            "sobel_y", sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} input channels, received {x.shape[1]}"
            )
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=self.channels)
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=self.channels)
        return grad_x, grad_y


class ComplexShapeExpert(nn.Module):
    """Figure-5 deformable-convolution and Sobel complex-shape expert."""

    def __init__(self, in_channels: int = 128, out_channels: int = 64, reduction: int = 16):
        super().__init__()
        if DeformConv2d is None:
            raise RuntimeError(
                "MoESteelUNetLast requires torchvision.ops.DeformConv2d, but it "
                f"could not be imported: {_DEFORM_CONV_IMPORT_ERROR}"
            )
        hidden = max(1, out_channels // reduction)
        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        self.deform_conv = DeformConv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.sobel = FixedSobelDepthwise(out_channels)
        self.boundary_fusion = _conv_bn_relu(
            out_channels * 2, out_channels, kernel_size=3, padding=1
        )
        self.shape_context = nn.Sequential(
            nn.Conv2d(out_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, deepest_feature: torch.Tensor) -> torch.Tensor:
        offsets = self.offset_conv(deepest_feature)
        deformed = self.deform_conv(deepest_feature, offsets)
        grad_x, grad_y = self.sobel(deformed)
        boundary = self.boundary_fusion(torch.cat([grad_x, grad_y], dim=1))
        return boundary * self.shape_context(boundary)


class GatingNetwork(nn.Module):
    """GAP/MLP sample-level router over the experts present in this variant."""

    def __init__(
        self,
        input_channels: int = 128,
        hidden_channels: int = 64,
        num_experts: int = len(EXPERT_ORDER),
    ):
        super().__init__()
        if num_experts < 2:
            raise ValueError("GatingNetwork is only needed for two or more experts")
        intermediate = hidden_channels * 2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, intermediate),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )
        self.router = nn.Linear(hidden_channels, num_experts)

    def forward(self, deepest_feature: torch.Tensor) -> torch.Tensor:
        descriptor = self.pool(deepest_feature).flatten(1)
        return F.softmax(self.router(self.mlp(descriptor)), dim=1)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, bilinear: bool):
        super().__init__()
        self.bilinear = bilinear
        self.up = None if bilinear else nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.bilinear:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        else:
            x = self.up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
        return self.conv(torch.cat([skip, x], dim=1))


class UNetDecoder(nn.Module):
    def __init__(self, expert_channels: int, num_classes: int, bilinear: bool = True):
        super().__init__()
        self.bottleneck = DoubleConv(expert_channels + 128, 128)
        self.up1 = UpBlock(128, 64, 64, bilinear)
        self.up2 = UpBlock(64, 32, 32, bilinear)
        self.up3 = UpBlock(32, 16, 16, bilinear)
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def decode_stages(
        self, fusion: torch.Tensor, skip_connections: Sequence[torch.Tensor]
    ):
        x1, x2, x3, x4 = skip_connections
        stage1 = self.bottleneck(torch.cat([fusion, x4], dim=1))
        stage2 = self.up1(stage1, x3)
        stage3 = self.up2(stage2, x2)
        stage4 = self.up3(stage3, x1)
        return stage1, stage2, stage3, stage4

    def forward(
        self, fusion: torch.Tensor, skip_connections: Sequence[torch.Tensor]
    ):
        stages = self.decode_stages(fusion, skip_connections)
        return self.classifier(stages[-1]), stages


def expert_specialization_loss(
    gate_weights: torch.Tensor,
    target_expert: torch.Tensor,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Mean ``1 - w_target`` over samples with an explicit expert target."""

    if gate_weights.ndim != 2 or gate_weights.shape[1] != len(EXPERT_ORDER):
        raise ValueError("gate_weights must have shape [batch, 3]")
    if target_expert.ndim != 1 or target_expert.shape[0] != gate_weights.shape[0]:
        raise ValueError("target_expert must have shape [batch]")
    valid = target_expert != ignore_index
    if not torch.any(valid):
        return gate_weights.sum() * 0.0
    valid_targets = target_expert[valid]
    if torch.any(valid_targets < 0) or torch.any(valid_targets >= len(EXPERT_ORDER)):
        raise ValueError("target expert indices must be 0=sde, 1=rde, or 2=cse")
    selected = gate_weights[valid].gather(1, valid_targets[:, None]).squeeze(1)
    return (1.0 - selected).mean()


class MoESteelUNetLast(nn.Module):
    """Figure-5 MoESteelUNet with a selectable non-empty expert subset."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        bilinear: bool = True,
        gate_hidden_channels: int = 64,
        active_experts: Sequence[str] = EXPERT_ORDER,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.gate_hidden_channels = gate_hidden_channels
        requested_experts = tuple(active_experts)
        unknown_experts = set(requested_experts) - set(EXPERT_ORDER)
        if unknown_experts:
            raise ValueError(
                f"Unknown experts: {sorted(unknown_experts)}. Expected: {EXPERT_ORDER}"
            )
        requested_set = set(requested_experts)
        if not requested_set:
            raise ValueError("active_experts must contain at least one expert")
        # Canonical order makes checkpoint metadata and report columns stable.
        self.active_experts = tuple(
            name for name in EXPERT_ORDER if name in requested_set
        )
        self.num_experts = len(self.active_experts)

        self.encoder = SharedEncoder(in_channels)
        expert_builders = {
            "sde": lambda: SmallDefectExpert(128, 64),
            "rde": lambda: RegionalDefectExpert(128, 64),
            "cse": lambda: ComplexShapeExpert(128, 64),
        }
        self.experts = nn.ModuleDict(
            {name: expert_builders[name]() for name in self.active_experts}
        )
        self.gating_network = (
            GatingNetwork(128, gate_hidden_channels, self.num_experts)
            if self.num_experts > 1
            else None
        )
        self.decoder = UNetDecoder(64, num_classes, bilinear)

    def get_model_config(self) -> Dict[str, object]:
        return {
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "bilinear": self.bilinear,
            "gate_hidden_channels": self.gate_hidden_channels,
            "active_experts": list(self.active_experts),
            "model_variant": "moesteelunet_last",
            "cse_variant": "figure5_deform_sobel",
            "deform_conv_variant": "unmodulated_dcnv1",
            "sobel_variant": "fixed_depthwise_xy_concat",
        }

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        deepest, skip_connections = self.encoder(x)
        expert_outputs = {
            name: self.experts[name](deepest) for name in self.active_experts
        }
        if self.num_experts == 1:
            fusion = next(iter(expert_outputs.values()))
            gate_weights = fusion.new_ones((fusion.shape[0], 1))
        else:
            gate_weights = self.gating_network(deepest)
            stacked = torch.stack(
                [expert_outputs[name] for name in self.active_experts], dim=1
            )
            fusion = (
                stacked * gate_weights[:, :, None, None, None]
            ).sum(dim=1)
        logits, decoder_stages = self.decoder(fusion, skip_connections)

        if not return_aux:
            return logits
        aux = {
            "active_experts": list(self.active_experts),
            "gate_weights": gate_weights,
            "encoder": deepest,
            "skip_connections": skip_connections,
            "fusion": fusion,
            "decoder_feature": decoder_stages[-1],
            "decoder_stages": decoder_stages,
        }
        aux.update(expert_outputs)
        return logits, aux


MoESteelUnetLast = MoESteelUNetLast


__all__ = [
    "EXPERT_ORDER",
    "ComplexShapeExpert",
    "FixedSobelDepthwise",
    "MoESteelUNetLast",
    "MoESteelUnetLast",
    "MultiScaleEncoderBlock",
    "expert_specialization_loss",
]
