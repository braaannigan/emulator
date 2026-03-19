"""Small CNN baseline for layer thickness emulation."""

from .config import CnnThicknessConfig, load_cnn_thickness_config
from .pipeline import run_cnn_thickness_experiment

__all__ = [
    "CnnThicknessConfig",
    "load_cnn_thickness_config",
    "run_cnn_thickness_experiment",
]
