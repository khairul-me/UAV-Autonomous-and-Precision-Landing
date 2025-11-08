# models/critic.py
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.feature_extractor import DepthFeatureExtractor

class Critic(nn.Module):
    """
    Critic network for DPRL (Q-network)
    Estimates Q(s, a) value
    
    In privileged learning:
    - Training: Receives CLEAN depth images
    - Testing: Not used (only Actor is deployed)
    """
    
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128):
        super(Critic, self).__init__()
        
        # Feature extractor for depth images
        self.depth_extractor = DepthFeatureExtractor(output_dim=25)
        
        # MLP for Q-value estimation
        # Input: 25 (depth) + 8 (state) + 4 (action) = 37
        self.fc1 = nn.Linear(25 + state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output: Q-value
        
        self.activation = nn.LeakyReLU(0.01)
    
    def forward(self, depth_image, self_state, action):
        """
        Args:
            depth_image: [batch, 1, 80, 100] (CLEAN in training with privileged learning)
            self_state: [batch, 8]
            action: [batch, 4]
        Returns:
            q_value: [batch, 1]
        """
        # Extract depth features
        depth_features = self.depth_extractor(depth_image)  # [batch, 25]
        
        # Concatenate everything
        x = torch.cat([depth_features, self_state, action], dim=1)  # [batch, 37]
        
        # MLP layers
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        q_value = self.fc3(x)  # [batch, 1]
        
        return q_value

# Test
if __name__ == "__main__":
    critic = Critic()
    
    # Dummy inputs
    depth = torch.randn(2, 1, 80, 100)
    state = torch.randn(2, 8)
    action = torch.randn(2, 4)
    
    q_value = critic(depth, state, action)
    
    print(f"Depth input: {depth.shape}")
    print(f"State input: {state.shape}")
    print(f"Action input: {action.shape}")
    print(f"Q-value output: {q_value.shape}")
    print(f"[OK] Critic network works!")

