"""Residual CNN emulator for layer thickness forecasting."""

from .config import ResidualThicknessConfig, load_residual_thickness_config
from .pipeline import run_residual_thickness_experiment

__all__ = [
    "ResidualThicknessConfig",
    "load_residual_thickness_config",
    "run_residual_thickness_experiment",
]
