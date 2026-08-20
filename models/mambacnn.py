"""CNN-primary encoder with a gated residual Mamba-1 context branch.

The original :class:`MultiScaleEncoderBlock` is kept as the complete local
path.  Mamba processes a reduced-channel view of that local feature and can
only add context through ``local + gamma * gate * global_feature``.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .moesteelunet_last import MultiScaleEncoderBlock
from mamba_ssm.modules.mamba_simple import Mamba


_VALID_SCAN_MODES = ("row_forward", "row_bidir", "axial_bidir")
_DEFAULT_GAMMA_INITS = (0.01, 0.025, 0.05, 0.10)


def _as_stage_tuple(name: str, values: Sequence[Any]) -> Tuple[Any, Any, Any, Any]:
    result = tuple(values)
    if len(result) != 4:
        raise ValueError("{} must contain exactly four stage values".format(name))
    return result  # type: ignore[return-value]


class AxialBidirectionalMamba(nn.Module):
    """Adapt one shared Mamba-1 instance to horizontal and vertical 2-D scans."""

    def __init__(
        self,
        channels: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 1,
        scan_mode: str = "axial_bidir",
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if scan_mode not in _VALID_SCAN_MODES:
            raise ValueError(
                "scan_mode must be one of {}, received {!r}".format(
                    _VALID_SCAN_MODES, scan_mode
                )
            )
        self.channels = channels
        self.scan_mode = scan_mode
        self.norm = nn.LayerNorm(channels)
        # The installed Mamba-1 implementation exposes this exact interface.
        # The explicit slow path avoids environment-dependent fused-kernel
        # selection and leaves site-packages untouched.
        self.mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )

    def _scan(self, sequence: torch.Tensor, bidirectional: bool) -> torch.Tensor:
        forward = self.mamba(sequence)
        if not bidirectional:
            return forward
        backward = torch.flip(
            self.mamba(torch.flip(sequence, dims=(1,))), dims=(1,)
        )
        return 0.5 * (forward + backward)

    def _scan_maps(
        self, channel_last: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Scan BHWC features and restore both maps to their original order."""
        if channel_last.ndim != 4 or channel_last.shape[-1] != self.channels:
            raise ValueError(
                "expected BHWC input with {} channels".format(self.channels)
            )
        batch, height, width, channels = channel_last.shape
        row_sequence = channel_last.reshape(batch * height, width, channels)
        horizontal = self._scan(
            row_sequence, bidirectional=self.scan_mode != "row_forward"
        ).reshape(batch, height, width, channels)

        vertical = None
        if self.scan_mode == "axial_bidir":
            column_sequence = (
                channel_last.permute(0, 2, 1, 3)
                .contiguous()
                .reshape(batch * width, height, channels)
            )
            vertical = (
                self._scan(column_sequence, bidirectional=True)
                .reshape(batch, width, height, channels)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
        return horizontal, vertical

    def forward(
        self, feature: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(
                "expected BCHW input with {} channels".format(self.channels)
            )
        if not feature.is_cuda:
            raise RuntimeError(
                "The installed Mamba-1 selective-scan implementation requires CUDA "
                "for forward execution; move the model and input to a CUDA device."
            )
        channel_last = feature.permute(0, 2, 3, 1).contiguous()
        horizontal, vertical = self._scan_maps(self.norm(channel_last))
        horizontal = horizontal.permute(0, 3, 1, 2).contiguous()
        if vertical is not None:
            vertical = vertical.permute(0, 3, 1, 2).contiguous()
        return horizontal, vertical


class CNNGuidedDirectionalFusion(nn.Module):
    """Predict horizontal/vertical mixture weights from the local CNN feature."""

    def __init__(self, local_channels: int):
        super().__init__()
        self.router = nn.Sequential(
            nn.Conv2d(
                local_channels,
                local_channels,
                kernel_size=3,
                padding=1,
                groups=local_channels,
                bias=False,
            ),
            nn.BatchNorm2d(local_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(local_channels, 2, kernel_size=1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(
        self,
        local: torch.Tensor,
        horizontal: torch.Tensor,
        vertical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if horizontal.shape != vertical.shape:
            raise ValueError("horizontal and vertical Mamba maps must have equal shapes")
        if local.shape[0] != horizontal.shape[0] or local.shape[-2:] != horizontal.shape[-2:]:
            raise ValueError("local and Mamba features must share batch/spatial dimensions")
        weights = torch.softmax(self.router(local), dim=1)
        fused = (
            weights[:, 0:1] * horizontal + weights[:, 1:2] * vertical
        )
        return fused, weights


class LocalGlobalInjectionGate(nn.Module):
    """Spatial gate controlling only the added global residual."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(8, channels // 4)
        self.network = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self, local: torch.Tensor, global_feature: torch.Tensor
    ) -> torch.Tensor:
        if local.shape != global_feature.shape:
            raise ValueError("local and global features must have equal BCHW shapes")
        return torch.sigmoid(self.network(torch.cat((local, global_feature), dim=1)))


class MambaCNNEncoderBlock(nn.Module):
    """Original multi-scale CNN plus an optional gated Mamba residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_mamba: bool = False,
        mamba_ratio: float = 0.5,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 1,
        scan_mode: str = "axial_bidir",
        gamma_init: float = 0.1,
        use_direction_router: bool = True,
        use_injection_gate: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_mamba = bool(use_mamba)
        self.mamba_ratio = float(mamba_ratio)
        self.scan_mode = scan_mode
        self.use_direction_router = bool(use_direction_router)
        self.use_injection_gate = bool(use_injection_gate)
        self.local_branch = MultiScaleEncoderBlock(in_channels, out_channels)

        self.reduce = None
        self.mamba_adapter = None
        self.direction_fusion = None
        self.global_refine = None
        self.expand_proj = None
        self.injection_gate = None
        self.register_parameter("gamma", None)
        self.mamba_channels = 0
        if not self.use_mamba:
            return

        if scan_mode not in _VALID_SCAN_MODES:
            raise ValueError(
                "an enabled Mamba stage requires one of {}, received {!r}".format(
                    _VALID_SCAN_MODES, scan_mode
                )
            )
        if not 0.0 < self.mamba_ratio <= 1.0:
            raise ValueError("an enabled Mamba stage requires 0 < mamba_ratio <= 1")
        self.mamba_channels = max(1, int(round(out_channels * self.mamba_ratio)))
        self.reduce = nn.Sequential(
            nn.Conv2d(
                out_channels, self.mamba_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(self.mamba_channels),
            nn.SiLU(inplace=True),
        )
        self.mamba_adapter = AxialBidirectionalMamba(
            channels=self.mamba_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            scan_mode=scan_mode,
        )
        if scan_mode == "axial_bidir" and self.use_direction_router:
            self.direction_fusion = CNNGuidedDirectionalFusion(out_channels)
        self.global_refine = nn.Sequential(
            nn.Conv2d(
                self.mamba_channels,
                self.mamba_channels,
                kernel_size=3,
                padding=1,
                groups=self.mamba_channels,
                bias=False,
            ),
            nn.BatchNorm2d(self.mamba_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                self.mamba_channels,
                self.mamba_channels,
                kernel_size=1,
                bias=False,
            ),
        )
        self.expand_proj = nn.Sequential(
            nn.Conv2d(
                self.mamba_channels, out_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )
        if self.use_injection_gate:
            self.injection_gate = LocalGlobalInjectionGate(out_channels)
        self.gamma = nn.Parameter(
            torch.full((1, out_channels, 1, 1), float(gamma_init))
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        local = self.local_branch(x)
        if not self.use_mamba:
            if return_aux:
                return local, {
                    "local_feature": local,
                    "mamba_enabled": False,
                }
            return local

        reduced = self.reduce(local)
        horizontal, vertical = self.mamba_adapter(reduced)
        if vertical is None:
            global_small = horizontal
            direction_weights = local.new_ones(
                (local.shape[0], 1, local.shape[2], local.shape[3])
            )
        elif self.direction_fusion is None:
            global_small = 0.5 * (horizontal + vertical)
            direction_weights = local.new_full(
                (local.shape[0], 2, local.shape[2], local.shape[3]), 0.5
            )
        else:
            global_small, direction_weights = self.direction_fusion(
                local, horizontal, vertical
            )
        global_small = global_small + self.global_refine(global_small)
        global_feature = self.expand_proj(global_small)
        if self.injection_gate is None:
            gate = local.new_ones(
                (local.shape[0], 1, local.shape[2], local.shape[3])
            )
        else:
            gate = self.injection_gate(local, global_feature)
        residual = self.gamma * gate * global_feature
        output = local + residual
        if not return_aux:
            return output
        residual_norm = torch.linalg.vector_norm(residual.detach())
        local_norm = torch.linalg.vector_norm(local.detach()).clamp_min(1e-12)
        aux: Dict[str, Any] = {
            "mamba_enabled": True,
            "local_feature": local,
            "reduced_feature": reduced,
            "horizontal_feature": horizontal,
            "vertical_feature": vertical,
            "direction_weights": direction_weights,
            "global_feature": global_feature,
            "injection_gate": gate,
            "mamba_residual": residual,
            "contribution_ratio": residual_norm / local_norm,
            "gamma_mean": self.gamma.detach().mean(),
        }
        return output, aux


class MambaCNNSharedEncoder(nn.Module):
    """Four-stage shared encoder preserving the baseline channel contract."""

    def __init__(
        self,
        in_channels: int = 3,
        encoder_channels: Sequence[int] = (16, 32, 64, 128),
        mamba_stages: Sequence[bool] = (True, True, True, True),
        mamba_ratios: Sequence[float] = (0.25, 0.25, 0.50, 0.50),
        scan_modes: Sequence[str] = (
            "axial_bidir",
            "axial_bidir",
            "axial_bidir",
            "axial_bidir",
        ),
        d_states: Sequence[int] = (4, 4, 8, 8),
        mamba_d_conv: int = 3,
        mamba_expand: int = 1,
        gamma_init: Optional[float] = None,
        gamma_inits: Optional[Sequence[float]] = None,
        use_direction_router: bool = True,
        use_injection_gate: bool = True,
    ):
        super().__init__()
        channels = _as_stage_tuple("encoder_channels", encoder_channels)
        if any(int(channel) <= 0 for channel in channels):
            raise ValueError("encoder_channels must be positive")
        self.encoder_channels = tuple(int(channel) for channel in channels)
        self.mamba_stages = tuple(
            bool(value) for value in _as_stage_tuple("mamba_stages", mamba_stages)
        )
        self.mamba_ratios = tuple(
            float(value) for value in _as_stage_tuple("mamba_ratios", mamba_ratios)
        )
        self.scan_modes = tuple(
            str(value) for value in _as_stage_tuple("scan_modes", scan_modes)
        )
        self.d_states = tuple(
            int(value) for value in _as_stage_tuple("d_states", d_states)
        )
        self.mamba_d_conv = int(mamba_d_conv)
        self.mamba_expand = int(mamba_expand)
        if gamma_inits is not None:
            self.gamma_inits = tuple(
                float(value)
                for value in _as_stage_tuple("gamma_inits", gamma_inits)
            )
            self.gamma_init = None
        elif gamma_init is not None:
            self.gamma_init = float(gamma_init)
            self.gamma_inits = (self.gamma_init,) * 4
        else:
            self.gamma_init = None
            self.gamma_inits = _DEFAULT_GAMMA_INITS
        self.use_direction_router = bool(use_direction_router)
        self.use_injection_gate = bool(use_injection_gate)
        self.pool = nn.MaxPool2d(2)

        inputs = (in_channels,) + self.encoder_channels[:-1]
        blocks = []
        for index, (stage_in, stage_out) in enumerate(
            zip(inputs, self.encoder_channels)
        ):
            blocks.append(
                MambaCNNEncoderBlock(
                    stage_in,
                    stage_out,
                    use_mamba=self.mamba_stages[index],
                    mamba_ratio=self.mamba_ratios[index],
                    d_state=self.d_states[index],
                    d_conv=self.mamba_d_conv,
                    expand=self.mamba_expand,
                    scan_mode=self.scan_modes[index],
                    gamma_init=self.gamma_inits[index],
                    use_direction_router=self.use_direction_router,
                    use_injection_gate=self.use_injection_gate,
                )
            )
        self.stage1, self.stage2, self.stage3, self.stage4 = blocks

    def get_config(self) -> Dict[str, Any]:
        return {
            "encoder_channels": list(self.encoder_channels),
            "mamba_stages": list(self.mamba_stages),
            "mamba_ratios": list(self.mamba_ratios),
            "scan_modes": list(self.scan_modes),
            "d_states": list(self.d_states),
            "mamba_d_conv": self.mamba_d_conv,
            "mamba_expand": self.mamba_expand,
            "encoder_variant": "four_stage_lpdm",
            "gamma_init": self.gamma_init,
            "gamma_inits": list(self.gamma_inits),
            "use_direction_router": self.use_direction_router,
            "use_injection_gate": self.use_injection_gate,
        }

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        if x.ndim != 4:
            raise ValueError("encoder input must be a BCHW tensor")
        features = []
        mamba_aux: Dict[str, Dict[str, Any]] = {}
        current = x
        for index, block in enumerate(
            (self.stage1, self.stage2, self.stage3, self.stage4), start=1
        ):
            if index > 1:
                current = self.pool(current)
            if return_aux:
                current, block_aux = block(current, return_aux=True)
                if block.use_mamba:
                    mamba_aux["stage{}".format(index)] = block_aux
            else:
                current = block(current)
            features.append(current)
        result = (features[-1], tuple(features))
        if not return_aux:
            return result
        return result[0], result[1], {
            "stage_features": tuple(features),
            "mamba_stages": mamba_aux,
        }


__all__ = [
    "AxialBidirectionalMamba",
    "CNNGuidedDirectionalFusion",
    "LocalGlobalInjectionGate",
    "MambaCNNEncoderBlock",
    "MambaCNNSharedEncoder",
]
