import torch
import torch.nn as nn


class Critic(nn.Module):
    """Twin Q-network for TD3."""

    def __init__(
        self,
        depth_feature_dim: int = 25,
        state_dim: int = 8,
        action_dim: int = 4,
    ) -> None:
        super().__init__()
        input_dim = depth_feature_dim + state_dim + action_dim

        self.q1_fc1 = nn.Linear(input_dim, 128)
        self.q1_fc2 = nn.Linear(128, 128)
        self.q1_fc3 = nn.Linear(128, 1)

        self.q2_fc1 = nn.Linear(input_dim, 128)
        self.q2_fc2 = nn.Linear(128, 128)
        self.q2_fc3 = nn.Linear(128, 1)

        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(
        self,
        depth_features: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([depth_features, state, action], dim=1)

        q1 = self.activation(self.q1_fc1(x))
        q1 = self.activation(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)

        q2 = self.activation(self.q2_fc1(x))
        q2 = self.activation(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)

        return q1, q2

    def q1(
        self,
        depth_features: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([depth_features, state, action], dim=1)
        q1 = self.activation(self.q1_fc1(x))
        q1 = self.activation(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        return q1

