# models/actor.py
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.feature_extractor import DepthFeatureExtractor

class Actor(nn.Module):
    """
    Actor network for DPRL
    Maps observations to actions
    """
    
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128):
        super(Actor, self).__init__()
        
        # Feature extractor for depth images
        self.depth_extractor = DepthFeatureExtractor(output_dim=25)
        
        # MLP for decision making
        # Input: 25 (depth features) + 8 (self-state) = 33
        self.fc1 = nn.Linear(25 + state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Use Leaky ReLU (from DPRL paper - prevents boundary outputs)
        self.activation = nn.LeakyReLU(0.01)
        
        # Output activation (tanh to bound actions)
        self.tanh = nn.Tanh()
        
        # Action scaling (from environment action bounds)
        self.action_scale = torch.FloatTensor([3.0, 3.0, 2.0, 0.3])
        self.action_bias = torch.FloatTensor([0.0, 0.0, 0.0, 0.0])
    
    def forward(self, depth_image, self_state):
        """
        Args:
            depth_image: [batch, 1, 80, 100]
            self_state: [batch, 8]
        Returns:
            action: [batch, 4]
        """
        # Extract depth features
        depth_features = self.depth_extractor(depth_image)  # [batch, 25]
        
        # Concatenate with self-state
        x = torch.cat([depth_features, self_state], dim=1)  # [batch, 33]
        
        # MLP layers
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        
        # Scale to action bounds
        action = self.tanh(x)  # [-1, 1]
        
        # Move action_scale to same device as action
        if self.action_scale.device != action.device:
            self.action_scale = self.action_scale.to(action.device)
            self.action_bias = self.action_bias.to(action.device)
        
        action = action * self.action_scale + self.action_bias
        
        return action

# Test
if __name__ == "__main__":
    actor = Actor()
    
    # Dummy inputs
    depth = torch.randn(2, 1, 80, 100)
    state = torch.randn(2, 8)
    
    action = actor(depth, state)
    
    print(f"Depth input: {depth.shape}")
    print(f"State input: {state.shape}")
    print(f"Action output: {action.shape}")
    print(f"Action range: [{action.min().item():.2f}, {action.max().item():.2f}]")
    print(f"[OK] Actor network works!")

