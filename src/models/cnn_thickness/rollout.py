from __future__ import annotations

import numpy as np
import torch

from .data import Standardizer


def autoregressive_rollout(
    model: torch.nn.Module,
    eval_frames: np.ndarray,
    standardizer: Standardizer,
    device: torch.device,
) -> np.ndarray:
    if eval_frames.shape[0] < 2:
        raise ValueError("Evaluation segment must contain at least two timesteps.")

    model.eval()
    current = standardizer.normalize(eval_frames[0]).astype(np.float32)
    rollout_frames: list[np.ndarray] = []

    with torch.no_grad():
        for _ in range(eval_frames.shape[0] - 1):
            input_tensor = torch.from_numpy(current).unsqueeze(0).unsqueeze(0).to(device)
            prediction = model(input_tensor).squeeze(0).squeeze(0).cpu().numpy()
            rollout_frames.append(standardizer.denormalize(prediction))
            current = prediction.astype(np.float32)

    return np.stack(rollout_frames, axis=0)
