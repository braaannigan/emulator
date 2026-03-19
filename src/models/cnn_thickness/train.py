from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import CnnThicknessConfig
from .data import AutoregressiveThicknessDataset


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_model(config: CnnThicknessConfig, model: torch.nn.Module, normalized_train_frames: np.ndarray) -> dict[str, float]:
    set_random_seed(config.random_seed)
    device = select_device()
    model.to(device)

    dataset = AutoregressiveThicknessDataset(normalized_train_frames.astype(np.float32))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()
    final_loss = 0.0
    optimization_steps = 0

    model.train()
    for _ in range(config.epochs):
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
            optimization_steps += 1

    return {
        "train_loss": final_loss,
        "device": str(device),
        "optimization_steps": optimization_steps,
        "training_examples": len(dataset),
    }
