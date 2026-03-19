from __future__ import annotations

import torch.nn as nn


class CnnThicknessModel(nn.Module):
    def __init__(self, hidden_channels: int, num_layers: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(1, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        ]
        for _ in range(max(num_layers - 2, 0)):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
                    nn.ReLU(),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=kernel_size, padding=padding))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)
