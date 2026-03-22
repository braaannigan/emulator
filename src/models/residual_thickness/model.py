from __future__ import annotations

import torch
import torch.nn as nn


def _build_normalization(normalization: str, channels: int) -> nn.Module:
    if normalization == "none":
        return nn.Identity()
    if normalization == "group":
        for groups in (8, 4, 2, 1):
            if channels % groups == 0:
                return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported normalization: {normalization}")


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        block_type: str,
        normalization: str,
    ):
        super().__init__()
        padding = (kernel_size // 2) * dilation

        if block_type == "standard":
            conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
            conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        elif block_type == "depthwise_separable":
            conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    groups=channels,
                ),
                nn.Conv2d(channels, channels, kernel_size=1),
            )
            conv2 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    groups=channels,
                ),
                nn.Conv2d(channels, channels, kernel_size=1),
            )
        else:
            raise ValueError(f"Unsupported block type: {block_type}")

        self.block = nn.Sequential(
            conv1,
            _build_normalization(normalization, channels),
            nn.GELU(),
            conv2,
            _build_normalization(normalization, channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class ResidualThicknessModel(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_blocks: int,
        kernel_size: int,
        block_type: str = "standard",
        normalization: str = "none",
        dilation_cycle: int = 1,
    ):
        super().__init__()
        self.input_projection = nn.Conv2d(1, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        blocks: list[nn.Module] = []
        cycle = max(dilation_cycle, 1)
        for block_index in range(num_blocks):
            dilation = 2 ** (block_index % cycle)
            blocks.append(
                ResidualBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    block_type=block_type,
                    normalization=normalization,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.output_projection = nn.Conv2d(hidden_channels, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs)
        hidden = self.blocks(hidden)
        return inputs + self.output_projection(hidden)
