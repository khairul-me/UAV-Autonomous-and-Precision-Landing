from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from attacks.base_attack import AdversarialAttack


class FGSM(AdversarialAttack):
    """Fast Gradient Sign Method."""

    def __init__(self, model, epsilon: float = 0.03, device: str = "cuda") -> None:
        super().__init__(model, device)
        self.epsilon = epsilon

    def attack(
        self, depth_image: np.ndarray, state: np.ndarray, target: np.ndarray | None = None
    ) -> np.ndarray:
        depth_tensor = self._to_tensor(depth_image)
        depth_tensor.requires_grad = True

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        features = self.model.feature_extractor(depth_tensor)
        action = self.model.actor(features, state_tensor)

        if target is None:
            loss = -torch.var(action)
        else:
            target_tensor = torch.FloatTensor(target).unsqueeze(0).to(self.device)
            loss = nn.functional.mse_loss(action, target_tensor)

        self.model.actor.zero_grad()
        self.model.feature_extractor.zero_grad()
        loss.backward()

        gradient = depth_tensor.grad.sign()
        adversarial_depth = depth_tensor + self.epsilon * gradient
        adversarial_depth = torch.clamp(adversarial_depth, 0, 100)

        return self._to_numpy(adversarial_depth)

