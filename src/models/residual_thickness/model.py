from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        input_channels: int,
        hidden_channels: int,
        num_blocks: int,
        kernel_size: int,
        model_variant: str = "residual",
        block_type: str = "standard",
        normalization: str = "none",
        dilation_cycle: int = 1,
        prognostic_channels: int = 1,
        state_history: int = 1,
        forcing_channels: int = 0,
        forcing_integration: str = "concat",
        transport_displacement_scale: float = 1.0,
        transport_correction_scale: float = 1.0,
        transport_head_mode: str = "shared",
    ):
        super().__init__()
        self.prognostic_channels = prognostic_channels
        self.state_history = state_history
        self.state_input_channels = prognostic_channels * max(state_history, 1)
        self.forcing_channels = forcing_channels
        self.forcing_integration = forcing_integration
        self.model_variant = model_variant
        self.transport_displacement_scale = transport_displacement_scale
        self.transport_correction_scale = transport_correction_scale
        self.transport_head_mode = transport_head_mode
        self.state_projection = None
        projection_channels = (
            input_channels
            if forcing_integration == "concat" or forcing_channels == 0
            else self.state_input_channels
        )
        if forcing_integration == "dual_branch" and forcing_channels > 0:
            self.input_projection = None
            self.state_projection = nn.Conv2d(
                self.state_input_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        else:
            self.input_projection = nn.Conv2d(
                projection_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        self.forcing_projection = None
        self.forcing_scale = None
        self.forcing_shift = None
        self.block_forcing_scales = None
        self.block_forcing_shifts = None
        self.forcing_encoder = None
        self.forcing_blocks = None
        self.fusion_projection = None
        if forcing_channels > 0 and forcing_integration == "add":
            self.forcing_projection = nn.Sequential(
                nn.Conv2d(forcing_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            )
        elif forcing_channels > 0 and forcing_integration == "film":
            self.forcing_scale = nn.Sequential(
                nn.Conv2d(forcing_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            )
            self.forcing_shift = nn.Sequential(
                nn.Conv2d(forcing_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            )
        elif forcing_channels > 0 and forcing_integration == "film_per_block":
            self.block_forcing_scales = nn.ModuleList()
            self.block_forcing_shifts = nn.ModuleList()
        elif forcing_channels > 0 and forcing_integration == "dual_branch":
            self.forcing_encoder = nn.Conv2d(
                forcing_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        elif forcing_integration not in {"concat", "add", "film", "film_per_block", "dual_branch"}:
            raise ValueError(f"Unsupported forcing integration: {forcing_integration}")
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
            if self.block_forcing_scales is not None and self.block_forcing_shifts is not None:
                self.block_forcing_scales.append(
                    nn.Sequential(
                        nn.Conv2d(forcing_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                        nn.GELU(),
                        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                    )
                )
                self.block_forcing_shifts.append(
                    nn.Sequential(
                        nn.Conv2d(forcing_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                        nn.GELU(),
                        nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        if self.forcing_encoder is not None:
            forcing_blocks: list[nn.Module] = []
            for block_index in range(num_blocks):
                dilation = 2 ** (block_index % cycle)
                forcing_blocks.append(
                    ResidualBlock(
                        channels=hidden_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        block_type=block_type,
                        normalization=normalization,
                    )
                )
            self.forcing_blocks = nn.Sequential(*forcing_blocks)
            self.fusion_projection = nn.Sequential(
                nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1),
                nn.GELU(),
            )
        self.output_projection = None
        self.displacement_head = None
        self.correction_head = None
        if model_variant == "residual":
            output_channels = prognostic_channels
            self.output_projection = nn.Conv2d(hidden_channels, output_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        elif model_variant == "transport" and prognostic_channels == 1 and transport_head_mode == "shared":
            output_channels = 3
            self.output_projection = nn.Conv2d(hidden_channels, output_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        elif model_variant == "transport" and prognostic_channels == 1 and transport_head_mode == "separate":
            self.displacement_head = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Conv2d(hidden_channels, 2, kernel_size=1),
            )
            self.correction_head = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Conv2d(hidden_channels, 1, kernel_size=1),
            )
        else:
            raise ValueError(f"Unsupported model_variant/head mode: {model_variant}/{transport_head_mode}")

    def _warp_state(self, current_state: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = current_state.shape
        device = current_state.device
        dtype = current_state.dtype
        y_coords = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x_coords = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)

        scale_x = 2.0 / max(width - 1, 1)
        scale_y = 2.0 / max(height - 1, 1)
        normalized_displacement = torch.stack(
            (
                displacement[:, 0] * scale_x,
                displacement[:, 1] * scale_y,
            ),
            dim=-1,
        )
        sampling_grid = base_grid - normalized_displacement
        return F.grid_sample(current_state, sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        state_inputs = inputs[:, : self.state_input_channels]
        current_state = state_inputs[:, : self.prognostic_channels]
        forcing_inputs = inputs[:, self.state_input_channels : self.state_input_channels + self.forcing_channels]
        if self.forcing_encoder is not None and self.forcing_blocks is not None and self.fusion_projection is not None:
            hidden = self.state_projection(state_inputs)
            for block in self.blocks:
                hidden = block(hidden)
            forcing_hidden = self.forcing_blocks(self.forcing_encoder(forcing_inputs))
            hidden = self.fusion_projection(torch.cat([hidden, forcing_hidden], dim=1))
        else:
            projected_inputs = inputs if self.forcing_integration == "concat" or self.forcing_channels == 0 else state_inputs
            hidden = self.input_projection(projected_inputs)
        if self.forcing_projection is not None:
            hidden = hidden + self.forcing_projection(forcing_inputs)
        elif self.forcing_scale is not None and self.forcing_shift is not None:
            scale = torch.tanh(self.forcing_scale(forcing_inputs))
            shift = self.forcing_shift(forcing_inputs)
            hidden = hidden * (1.0 + scale) + shift
        if self.forcing_encoder is None:
            if self.block_forcing_scales is not None and self.block_forcing_shifts is not None:
                for block, scale_layer, shift_layer in zip(self.blocks, self.block_forcing_scales, self.block_forcing_shifts):
                    hidden = block(hidden)
                    scale = torch.tanh(scale_layer(forcing_inputs))
                    shift = shift_layer(forcing_inputs)
                    hidden = hidden * (1.0 + scale) + shift
            else:
                for block in self.blocks:
                    hidden = block(hidden)
        if self.model_variant == "transport":
            if self.displacement_head is not None and self.correction_head is not None:
                displacement = torch.tanh(self.displacement_head(hidden)) * self.transport_displacement_scale
                correction = torch.tanh(self.correction_head(hidden)) * self.transport_correction_scale
            else:
                outputs = self.output_projection(hidden)
                displacement = torch.tanh(outputs[:, :2]) * self.transport_displacement_scale
                correction = torch.tanh(outputs[:, 2:3]) * self.transport_correction_scale
            transported = self._warp_state(current_state, displacement)
            return transported + correction
        outputs = self.output_projection(hidden)
        return current_state + outputs
