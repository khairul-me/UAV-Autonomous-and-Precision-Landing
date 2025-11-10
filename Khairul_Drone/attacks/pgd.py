from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from attacks.base_attack import AdversarialAttack


class PGD(AdversarialAttack):
    """Projected Gradient Descent attack."""

    def __init__(
        self,
        model,
        epsilon: float = 0.03,
        alpha: float = 0.007,
        num_iter: int = 10,
        device: str = "cuda",
    ) -> None:
        super().__init__(model, device)
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter

    def attack(
        self, depth_image: np.ndarray, state: np.ndarray, target: np.ndarray | None = None
    ) -> np.ndarray:
        depth_orig = self._to_tensor(depth_image)
        adversarial_depth = (
            depth_orig + torch.randn_like(depth_orig) * 0.01
        ).clamp(depth_orig - self.epsilon, depth_orig + self.epsilon)
        adversarial_depth = torch.clamp(adversarial_depth, 0, 100)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        for _ in range(self.num_iter):
            adversarial_depth.requires_grad = True
            features = self.model.feature_extractor(adversarial_depth)
            action = self.model.actor(features, state_tensor)

            if target is None:
                loss = -torch.var(action)
            else:
                target_tensor = torch.FloatTensor(target).unsqueeze(0).to(self.device)
                loss = nn.functional.mse_loss(action, target_tensor)

            self.model.actor.zero_grad()
            self.model.feature_extractor.zero_grad()
            loss.backward()

            gradient = adversarial_depth.grad.sign()
            adversarial_depth = adversarial_depth.detach() + self.alpha * gradient

            perturbation = (adversarial_depth - depth_orig).clamp(
                -self.epsilon, self.epsilon
            )
            adversarial_depth = depth_orig + perturbation
            adversarial_depth = torch.clamp(adversarial_depth, 0, 100)

        return self._to_numpy(adversarial_depth)

