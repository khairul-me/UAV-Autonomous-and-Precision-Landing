import torch
import torch.nn as nn


class Actor(nn.Module):
    """Policy network that maps observations to actions."""

    def __init__(
        self,
        depth_feature_dim: int = 25,
        state_dim: int = 8,
        action_dim: int = 4,
        max_action: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_action = max_action
        input_dim = depth_feature_dim + state_dim

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, depth_features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([depth_features, state], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action * self.max_action

    def get_action(
        self, depth_features: torch.Tensor, state: torch.Tensor, noise: float = 0.0
    ) -> torch.Tensor:
        action = self.forward(depth_features, state)
        if noise > 0:
            action = action + torch.randn_like(action) * noise
            action = torch.clamp(action, -self.max_action, self.max_action)
        return action

