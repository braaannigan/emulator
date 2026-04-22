from __future__ import annotations

import torch
import torch.nn as nn


def _validate_padding_mode(boundary_padding_mode: str) -> str:
    allowed = {"zeros", "reflect", "replicate", "circular"}
    if boundary_padding_mode not in allowed:
        raise ValueError(f"Unsupported boundary_padding_mode: {boundary_padding_mode}")
    return boundary_padding_mode


def _build_norm(norm_type: str, channels: int) -> nn.Module:
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "groupnorm":
        return nn.GroupNorm(1, channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_type: str,
        dilation: int,
        boundary_padding_mode: str,
    ):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                padding_mode=boundary_padding_mode,
            ),
            _build_norm(norm_type, out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                padding_mode=boundary_padding_mode,
            ),
            _build_norm(norm_type, out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, norm_type: str, dilation: int, boundary_padding_mode: str):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        expanded_channels = channels * 4
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            dilation=dilation,
            padding_mode=boundary_padding_mode,
        )
        self.norm = _build_norm(norm_type, channels)
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


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, norm_type: str, dilation: int, boundary_padding_mode: str):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                dilation=dilation,
                padding_mode=boundary_padding_mode,
            ),
            _build_norm(norm_type, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            _build_norm(norm_type, channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                dilation=dilation,
                padding_mode=boundary_padding_mode,
            ),
            _build_norm(norm_type, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            _build_norm(norm_type, channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class FlexibleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        block_type: str,
        dilations: tuple[int, ...],
        norm_type: str,
        boundary_padding_mode: str,
    ):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        repeated_blocks: list[nn.Module] = []
        for dilation in dilations:
            if block_type == "standard":
                repeated_blocks.append(
                    ConvBlock(out_channels, out_channels, kernel_size, norm_type, dilation, boundary_padding_mode)
                )
            elif block_type == "convnext":
                repeated_blocks.append(
                    ConvNeXtBlock(out_channels, kernel_size, norm_type, dilation, boundary_padding_mode)
                )
            elif block_type == "depthwise_separable":
                repeated_blocks.append(
                    DepthwiseSeparableBlock(out_channels, kernel_size, norm_type, dilation, boundary_padding_mode)
                )
            else:
                raise ValueError(f"Unsupported block_type: {block_type}")
        self.block = nn.Sequential(*repeated_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(self.projection(inputs))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample_mode: str):
        super().__init__()
        self.upsample_mode = upsample_mode
        if upsample_mode == "transpose":
            self.projection = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif upsample_mode == "bilinear":
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            raise ValueError(f"Unsupported upsample_mode: {upsample_mode}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.upsample_mode == "bilinear":
            inputs = nn.functional.interpolate(inputs, scale_factor=2, mode="bilinear", align_corners=False)
        return self.projection(inputs)


class UnetThicknessModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_levels: int = 2,
        kernel_size: int = 3,
        stage_depth: int = 1,
        dilation_cycle: int = 1,
        norm_type: str = "groupnorm",
        state_channels: int = 1,
        output_steps: int = 1,
        forcing_channels: int = 0,
        fusion_mode: str = "input",
        skip_fusion_mode: str = "concat",
        upsample_mode: str = "transpose",
        residual_connection: bool = True,
        residual_step_scale: float = 1.0,
        block_type: str = "standard",
        prognostic_channels: int = 1,
        boundary_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.state_channels = state_channels
        self.prognostic_channels = prognostic_channels
        self.forcing_channels = forcing_channels
        self.output_steps = max(output_steps, 1)
        self.fusion_mode = fusion_mode
        self.skip_fusion_mode = skip_fusion_mode
        self.residual_connection = residual_connection
        self.residual_step_scale = residual_step_scale
        self.boundary_padding_mode = _validate_padding_mode(boundary_padding_mode)
        levels = max(num_levels, 1)
        dilation_cycle = max(dilation_cycle, 1)

        block_counter = 0

        def next_dilations() -> tuple[int, ...]:
            nonlocal block_counter
            stage_dilations: list[int] = []
            for _ in range(max(stage_depth, 1)):
                stage_dilations.append(2 ** (block_counter % dilation_cycle))
                block_counter += 1
            return tuple(stage_dilations)

        encoder_channels = [hidden_channels * (2**level) for level in range(levels)]
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        encoder_input_channels = input_channels if fusion_mode == "input" else state_channels
        in_channels = encoder_input_channels
        for out_channels in encoder_channels:
            self.encoders.append(
                FlexibleBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    block_type,
                    next_dilations(),
                    norm_type,
                    self.boundary_padding_mode,
                )
            )
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        bottleneck_channels = encoder_channels[-1] * 2
        self.bottleneck = FlexibleBlock(
            encoder_channels[-1],
            bottleneck_channels,
            kernel_size,
            block_type,
            next_dilations(),
            norm_type,
            self.boundary_padding_mode,
        )
        self.forcing_encoder = None
        self.forcing_scale_blocks = None
        self.forcing_scale_pools = None
        self.forcing_bottleneck = None
        if fusion_mode in {"bottleneck", "per_scale"} and forcing_channels > 0:
            self.forcing_scale_blocks = nn.ModuleList()
            self.forcing_scale_pools = nn.ModuleList()
            current_channels = forcing_channels
            for out_channels in encoder_channels:
                self.forcing_scale_blocks.append(
                    FlexibleBlock(
                        current_channels,
                        out_channels,
                        kernel_size,
                        block_type,
                        next_dilations(),
                        norm_type,
                        self.boundary_padding_mode,
                    )
                )
                self.forcing_scale_pools.append(nn.AvgPool2d(kernel_size=2, stride=2))
                current_channels = out_channels
            self.forcing_bottleneck = FlexibleBlock(
                encoder_channels[-1],
                bottleneck_channels,
                kernel_size,
                block_type,
                next_dilations(),
                norm_type,
                self.boundary_padding_mode,
            )

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        self.skip_gates = nn.ModuleList()
        current_channels = bottleneck_channels
        for skip_channels in reversed(encoder_channels):
            self.upconvs.append(UpsampleBlock(current_channels, skip_channels, upsample_mode))
            if skip_fusion_mode == "concat":
                decoder_in_channels = skip_channels * 2
                self.skip_projections.append(nn.Identity())
                self.skip_gates.append(nn.Identity())
            elif skip_fusion_mode == "add":
                decoder_in_channels = skip_channels
                self.skip_projections.append(nn.Identity())
                self.skip_gates.append(nn.Identity())
            elif skip_fusion_mode == "gated":
                decoder_in_channels = skip_channels * 2
                self.skip_projections.append(nn.Identity())
                self.skip_gates.append(nn.Conv2d(skip_channels * 2, skip_channels, kernel_size=1))
            else:
                raise ValueError(f"Unsupported skip_fusion_mode: {skip_fusion_mode}")
            self.decoders.append(
                FlexibleBlock(
                    decoder_in_channels,
                    skip_channels,
                    kernel_size,
                    block_type,
                    next_dilations(),
                    norm_type,
                    self.boundary_padding_mode,
                )
            )
            current_channels = skip_channels

        self.output_projection = nn.Conv2d(current_channels, prognostic_channels * self.output_steps, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        state_inputs = inputs[:, : self.state_channels]
        forcing_inputs = inputs[:, self.state_channels :]
        skips: list[torch.Tensor] = []
        hidden = inputs if self.fusion_mode == "input" else state_inputs
        forcing_hidden = forcing_inputs
        for level, (encoder, pool) in enumerate(zip(self.encoders, self.pools)):
            hidden = encoder(hidden)
            if self.fusion_mode == "per_scale" and self.forcing_scale_blocks is not None and forcing_hidden.shape[1] > 0:
                forcing_hidden = self.forcing_scale_blocks[level](forcing_hidden)
                hidden = hidden + forcing_hidden
            skips.append(hidden)
            hidden = pool(hidden)
            if self.fusion_mode in {"bottleneck", "per_scale"} and self.forcing_scale_pools is not None and forcing_hidden.shape[1] > 0:
                if self.fusion_mode == "bottleneck" and self.forcing_scale_blocks is not None:
                    forcing_hidden = self.forcing_scale_blocks[level](forcing_hidden)
                forcing_hidden = self.forcing_scale_pools[level](forcing_hidden)

        hidden = self.bottleneck(hidden)
        if self.forcing_bottleneck is not None and forcing_hidden.shape[1] > 0:
            hidden = hidden + self.forcing_bottleneck(forcing_hidden)

        for upconv, decoder, skip_projection, skip_gate, skip in zip(
            self.upconvs,
            self.decoders,
            self.skip_projections,
            self.skip_gates,
            reversed(skips),
        ):
            hidden = upconv(hidden)
            if hidden.shape[-2:] != skip.shape[-2:]:
                hidden = nn.functional.interpolate(hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            if self.skip_fusion_mode == "concat":
                hidden = torch.cat([hidden, skip_projection(skip)], dim=1)
            elif self.skip_fusion_mode == "add":
                hidden = hidden + skip_projection(skip)
            else:
                gated_skip = skip * torch.sigmoid(skip_gate(torch.cat([hidden, skip], dim=1)))
                hidden = torch.cat([hidden, gated_skip], dim=1)
            hidden = decoder(hidden)

        outputs = self.output_projection(hidden)
        outputs = outputs.view(
            outputs.shape[0],
            self.output_steps,
            self.prognostic_channels,
            outputs.shape[-2],
            outputs.shape[-1],
        )
        if self.residual_connection:
            residual_base = state_inputs[:, : self.prognostic_channels].unsqueeze(1)
            outputs = residual_base + (self.residual_step_scale * outputs)
        if self.output_steps == 1:
            return outputs[:, 0]
        return outputs
