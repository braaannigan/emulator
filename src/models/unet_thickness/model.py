from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        expanded_channels = channels * 4
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, expanded_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(expanded_channels, channels, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(inputs)
        hidden = self.norm(hidden)
        hidden = self.pointwise(hidden)
        return inputs + hidden


class FlexibleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, block_type: str):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        if block_type == "standard":
            self.block = ConvBlock(out_channels, out_channels, kernel_size)
        elif block_type == "convnext":
            self.block = nn.Sequential(
                ConvNeXtBlock(out_channels, kernel_size),
                ConvNeXtBlock(out_channels, kernel_size),
            )
        else:
            raise ValueError(f"Unsupported block_type: {block_type}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(self.projection(inputs))


class UnetThicknessModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_levels: int = 2,
        kernel_size: int = 3,
        state_channels: int = 1,
        forcing_channels: int = 0,
        fusion_mode: str = "input",
        residual_connection: bool = True,
        residual_step_scale: float = 1.0,
        block_type: str = "standard",
        prognostic_channels: int = 1,
    ):
        super().__init__()
        self.state_channels = state_channels
        self.prognostic_channels = prognostic_channels
        self.forcing_channels = forcing_channels
        self.fusion_mode = fusion_mode
        self.residual_connection = residual_connection
        self.residual_step_scale = residual_step_scale
        levels = max(num_levels, 1)

        encoder_channels = [hidden_channels * (2**level) for level in range(levels)]
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        encoder_input_channels = input_channels if fusion_mode == "input" else state_channels
        in_channels = encoder_input_channels
        for out_channels in encoder_channels:
            self.encoders.append(FlexibleBlock(in_channels, out_channels, kernel_size, block_type))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        bottleneck_channels = encoder_channels[-1] * 2
        self.bottleneck = FlexibleBlock(encoder_channels[-1], bottleneck_channels, kernel_size, block_type)
        self.forcing_encoder = None
        if fusion_mode == "bottleneck" and forcing_channels > 0:
            forcing_layers: list[nn.Module] = []
            current_channels = forcing_channels
            for out_channels in encoder_channels:
                forcing_layers.append(FlexibleBlock(current_channels, out_channels, kernel_size, block_type))
                forcing_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_channels = out_channels
            forcing_layers.append(FlexibleBlock(encoder_channels[-1], bottleneck_channels, kernel_size, block_type))
            self.forcing_encoder = nn.Sequential(*forcing_layers)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current_channels = bottleneck_channels
        for skip_channels in reversed(encoder_channels):
            self.upconvs.append(nn.ConvTranspose2d(current_channels, skip_channels, kernel_size=2, stride=2))
            self.decoders.append(FlexibleBlock(skip_channels * 2, skip_channels, kernel_size, block_type))
            current_channels = skip_channels

        self.output_projection = nn.Conv2d(current_channels, prognostic_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        state_inputs = inputs[:, : self.state_channels]
        forcing_inputs = inputs[:, self.state_channels :]
        skips: list[torch.Tensor] = []
        hidden = inputs if self.fusion_mode == "input" else state_inputs
        for encoder, pool in zip(self.encoders, self.pools):
            hidden = encoder(hidden)
            skips.append(hidden)
            hidden = pool(hidden)

        hidden = self.bottleneck(hidden)
        if self.forcing_encoder is not None:
            hidden = hidden + self.forcing_encoder(forcing_inputs)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            hidden = upconv(hidden)
            if hidden.shape[-2:] != skip.shape[-2:]:
                hidden = nn.functional.interpolate(hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            hidden = torch.cat([hidden, skip], dim=1)
            hidden = decoder(hidden)

        outputs = self.output_projection(hidden)
        if self.residual_connection:
            return state_inputs[:, : self.prognostic_channels] + (self.residual_step_scale * outputs)
        return outputs
