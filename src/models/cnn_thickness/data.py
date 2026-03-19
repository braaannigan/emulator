from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


@dataclass(frozen=True)
class Standardizer:
    mean: float
    std: float

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std) + self.mean


@dataclass(frozen=True)
class SplitData:
    train_frames: np.ndarray
    eval_frames: np.ndarray
    train_time_days: np.ndarray
    eval_time_days: np.ndarray


def load_field_dataset(netcdf_path: str, field_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = xr.open_dataset(netcdf_path)
    try:
        data_array = dataset[field_name]
        if "layers" in data_array.dims:
            data_array = data_array.sel(layers=0)
        values = np.asarray(data_array.values, dtype=np.float32)
        time_days = np.asarray(dataset["time_days"].values, dtype=np.float32)
        y = np.asarray(data_array[data_array.dims[-2]].values, dtype=np.float32)
        x = np.asarray(data_array[data_array.dims[-1]].values, dtype=np.float32)
    finally:
        dataset.close()
    return values, time_days, y, x


def split_sequence(frames: np.ndarray, time_days: np.ndarray, train_fraction: float) -> SplitData:
    total_steps = frames.shape[0]
    split_index = int(np.floor(total_steps * train_fraction))
    split_index = min(max(split_index, 2), total_steps - 2)
    return SplitData(
        train_frames=frames[:split_index],
        eval_frames=frames[split_index:],
        train_time_days=time_days[:split_index],
        eval_time_days=time_days[split_index:],
    )


def fit_standardizer(train_frames: np.ndarray) -> Standardizer:
    mean = float(train_frames.mean())
    std = float(train_frames.std())
    if std == 0.0:
        std = 1.0
    return Standardizer(mean=mean, std=std)


class AutoregressiveThicknessDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, normalized_frames: np.ndarray):
        self.inputs = normalized_frames[:-1]
        self.targets = normalized_frames[1:]

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_tensor = torch.from_numpy(self.inputs[index]).unsqueeze(0)
        target_tensor = torch.from_numpy(self.targets[index]).unsqueeze(0)
        return input_tensor, target_tensor
