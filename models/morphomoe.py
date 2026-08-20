"""MoESteelUNet variant using the four-stage LPDM shared encoder."""

from typing import Dict, Optional, Sequence

import torch.nn as nn

from .moesteelunet_last import EXPERT_ORDER, MoESteelUNetLast
from .mambacnn import MambaCNNSharedEncoder


class MoESteelUNetMambaCNN(MoESteelUNetLast):
    """Final MoESteelUNet with only its shared encoder replaced."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        bilinear: bool = True,
        gate_hidden_channels: int = 64,
        active_experts: Sequence[str] = EXPERT_ORDER,
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
        expected_channels = (16, 32, 64, 128)
        if tuple(encoder_channels) != expected_channels:
            raise ValueError(
                "encoder_channels must remain {} for the unchanged experts and "
                "decoder".format(expected_channels)
            )
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            bilinear=bilinear,
            gate_hidden_channels=gate_hidden_channels,
            active_experts=active_experts,
        )
        # Keep the reusable baseline MoE implementation unchanged while making
        # the final MorphoMoE SDE match Eq. (12): dilation 2 followed by
        # dilation 3, both with 3x3 kernels and BN+ReLU.
        if "sde" in self.experts:
            self.experts["sde"].dilated_branch[1] = nn.Sequential(
                nn.Conv2d(
                    64,
                    64,
                    kernel_size=3,
                    padding=3,
                    dilation=3,
                    bias=False,
                ),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        # Only the shared encoder is changed. Experts, gating network, decoder
        # and inherited forward behavior remain the final baseline objects.
        self.encoder = MambaCNNSharedEncoder(
            in_channels=in_channels,
            encoder_channels=encoder_channels,
            mamba_stages=mamba_stages,
            mamba_ratios=mamba_ratios,
            scan_modes=scan_modes,
            d_states=d_states,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            gamma_init=gamma_init,
            gamma_inits=gamma_inits,
            use_direction_router=use_direction_router,
            use_injection_gate=use_injection_gate,
        )

    def get_model_config(self) -> Dict[str, object]:
        config = super().get_model_config()
        config.update(self.encoder.get_config())
        config.update(
            {
                "model_variant": "moemambacnn",
                "encoder_variant": "four_stage_lpdm",
                "mamba_impl": "mamba_ssm.modules.mamba_simple.Mamba",
                "mamba_use_fast_path": False,
            }
        )
        return config


MoESteelUnetMambaCNN = MoESteelUNetMambaCNN
MorphoMoE = MoESteelUNetMambaCNN


__all__ = ["MorphoMoE", "MoESteelUNetMambaCNN", "MoESteelUnetMambaCNN"]
