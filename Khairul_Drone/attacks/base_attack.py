from __future__ import annotations

import numpy as np
import torch


class AdversarialAttack:
    """Base class for adversarial perturbations."""

    def __init__(self, model, device: str = "cuda") -> None:
        self.model = model
        self.device = device if torch.cuda.is_available() else "cpu"

    def attack(
        self, depth_image: np.ndarray, state: np.ndarray, target: np.ndarray | None = None
    ) -> np.ndarray:
        raise NotImplementedError

    def _to_tensor(self, depth: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(depth).unsqueeze(0).unsqueeze(0).to(self.device)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.squeeze().detach().cpu().numpy()

