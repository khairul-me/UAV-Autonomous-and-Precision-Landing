from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


class TD3:
    """Twin Delayed DDPG implementation."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        actor: nn.Module,
        critic: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if critic is None:
            raise ValueError("Critic network must be provided for TD3.")

        self.feature_extractor = feature_extractor.to(self.device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        self.feature_extractor_target = copy.deepcopy(self.feature_extractor)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(
        self, depth: np.ndarray, state: np.ndarray, noise: float = 0.0
    ) -> np.ndarray:
        import numpy as np

        self.feature_extractor.eval()
        self.actor.eval()

        with torch.no_grad():
            depth_tensor = (
                torch.FloatTensor(depth).unsqueeze(0).unsqueeze(0).to(self.device)
            )
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            features = self.feature_extractor(depth_tensor)
            action = self.actor(features, state_tensor)

            if noise > 0:
                noise_tensor = torch.randn_like(action) * noise
                action = action + noise_tensor
                action = torch.clamp(action, -1.0, 1.0)

            result = action.cpu().numpy()[0]

        self.feature_extractor.train()
        self.actor.train()
        return result

    def train(
        self, replay_buffer, batch_size: int = 128
    ) -> Tuple[float, Optional[float]]:
        self.total_it += 1

        batch = replay_buffer.sample(batch_size)

        depth = batch["depth"].unsqueeze(1).to(self.device)
        state = batch["state"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_depth = batch["next_depth"].unsqueeze(1).to(self.device)
        next_state = batch["next_state"].to(self.device)
        done = batch["done"].to(self.device)

        with torch.no_grad():
            next_features = self.feature_extractor_target(next_depth)
        features = self.feature_extractor(depth)

        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (
                self.actor_target(next_features, next_state) + noise
            ).clamp(-1.0, 1.0)

            target_q1, target_q2 = self.critic_target(
                next_features, next_state, next_action
            )
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(features.detach(), state, action)
        critic_loss = (
            nn.functional.mse_loss(current_q1, target_q)
            + nn.functional.mse_loss(current_q2, target_q)
        )

        self.critic_optimizer.zero_grad()
        self.feature_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.feature_optimizer.step()

        actor_loss_val: Optional[float] = None
        if self.total_it % self.policy_freq == 0:
            features = self.feature_extractor(depth)
            policy_action = self.actor(features, state)
            actor_loss = -self.critic.q1(features, state, policy_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.feature_extractor, self.feature_extractor_target)
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            actor_loss_val = actor_loss.item()

        return critic_loss.item(), actor_loss_val

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, filepath: str) -> None:
        torch.save(
            {
                "feature_extractor": self.feature_extractor.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])


