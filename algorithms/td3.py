# algorithms/td3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.actor import Actor
from models.critic import Critic

class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)
    Based on DPRL paper's training algorithm
    
    Key features:
    1. Two Critic networks (reduce overestimation)
    2. Delayed policy updates
    3. Target policy smoothing
    4. Privileged learning support
    """
    
    def __init__(
        self,
        state_dim=8,
        action_dim=4,
        max_action=None,  # [3.0, 3.0, 2.0, 0.3]
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        actor_lr=3e-4,
        critic_lr=3e-4,
        device='cuda'
    ):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Actor network and target
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Two Critic networks and targets (for clipped double Q-learning)
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        
        # Hyperparameters
        self.max_action = torch.FloatTensor(max_action).to(self.device) if max_action else None
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.total_it = 0
    
    def select_action(self, depth_image, state, add_noise=True, noise_scale=0.1):
        """
        Select action using Actor network
        
        Args:
            depth_image: numpy array [80, 100]
            state: numpy array [8]
            add_noise: Whether to add exploration noise
            noise_scale: Scale of exploration noise
        """
        # Convert to torch tensors
        depth = torch.FloatTensor(depth_image).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 80, 100]
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, 8]
        
        # Get action
        with torch.no_grad():
            action = self.actor(depth, state).cpu().numpy()[0]
        
        # Add exploration noise
        if add_noise:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = action + noise
            
            # Clip to action bounds
            if self.max_action is not None:
                max_action = self.max_action.cpu().numpy()
                action = np.clip(action, -max_action, max_action)
        
        return action
    
    def train(self, replay_buffer, batch_size=128, use_privileged=False):
        """
        Train TD3 for one step
        
        Args:
            replay_buffer: ReplayBuffer object
            batch_size: Batch size for training
            use_privileged: If True, use clean depth for Critic (privileged learning)
        
        Returns:
            Dictionary with training metrics
        """
        self.total_it += 1
        
        # Sample replay buffer
        batch = replay_buffer.sample(batch_size, use_privileged=use_privileged)
        
        # Move to device
        depth = batch['depth'].unsqueeze(1).to(self.device)  # [B, 1, 80, 100]
        state = batch['state'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_depth = batch['next_depth'].unsqueeze(1).to(self.device)
        next_state = batch['next_state'].to(self.device)
        done = batch['done'].to(self.device)
        
        # For Critic: use clean depth if privileged learning
        if use_privileged:
            critic_depth = batch['clean_depth'].unsqueeze(1).to(self.device)
            critic_next_depth = batch['next_clean_depth'].unsqueeze(1).to(self.device)
        else:
            critic_depth = depth
            critic_next_depth = next_depth
        
        # ==================== Update Critics ====================
        
        with torch.no_grad():
            # Select next action from target Actor
            next_action = self.actor_target(next_depth, next_state)
            
            # Add clipped noise (target policy smoothing)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = next_action + noise
            
            # Clip to action bounds
            if self.max_action is not None:
                next_action = torch.clamp(next_action, -self.max_action, self.max_action)
            
            # Compute target Q-values (clipped double Q-learning)
            target_Q1 = self.critic_1_target(critic_next_depth, next_state, next_action)
            target_Q2 = self.critic_2_target(critic_next_depth, next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q
        
        # Get current Q-value estimates
        current_Q1 = self.critic_1(critic_depth, state, action)
        current_Q2 = self.critic_2(critic_depth, state, action)
        
        # Compute Critic losses
        critic_loss_1 = F.mse_loss(current_Q1, target_Q)
        critic_loss_2 = F.mse_loss(current_Q2, target_Q)
        
        # Update Critic 1
        self.critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic_1_optimizer.step()
        
        # Update Critic 2
        self.critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        self.critic_2_optimizer.step()
        
        # ==================== Delayed Actor Update ====================
        
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            
            # Compute Actor loss (maximize Q-value)
            actor_action = self.actor(depth, state)
            actor_loss = -self.critic_1(critic_depth, state, actor_action).mean()
            
            # Update Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic_1, self.critic_1_target)
            self._soft_update(self.critic_2, self.critic_2_target)
        
        # Return metrics
        metrics = {
            'critic_loss_1': critic_loss_1.item(),
            'critic_loss_2': critic_loss_2.item(),
            'actor_loss': actor_loss.item() if actor_loss is not None else None,
            'q_value': current_Q1.mean().item()
        }
        
        return metrics
    
    def _soft_update(self, source, target):
        """Soft update target network parameters: θ_target = τ*θ_source + (1-τ)*θ_target"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filename):
        """Save model"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
        }, filename)
    
    def load(self, filename):
        """Load model"""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer'])

# Test
if __name__ == "__main__":
    import numpy as np
    from algorithms.replay_buffer import ReplayBuffer
    
    # Initialize
    agent = TD3(max_action=[3.0, 3.0, 2.0, 0.3])
    buffer = ReplayBuffer()
    
    # Fill buffer with dummy data
    print("Filling buffer with dummy data...")
    for i in range(500):
        depth = np.random.randn(80, 100).astype(np.float32)
        state = np.random.randn(8).astype(np.float32)
        action = np.random.randn(4).astype(np.float32)
        reward = np.random.randn()
        next_depth = np.random.randn(80, 100).astype(np.float32)
        next_state = np.random.randn(8).astype(np.float32)
        done = 0.0
        
        buffer.add(depth, state, action, reward, next_depth, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Test training
    print("\nTesting TD3 training...")
    metrics = agent.train(buffer, batch_size=32)
    print(f"✓ Critic loss 1: {metrics['critic_loss_1']:.4f}")
    print(f"✓ Critic loss 2: {metrics['critic_loss_2']:.4f}")
    print(f"✓ Q-value: {metrics['q_value']:.4f}")
    
    # Test action selection
    print("\nTesting action selection...")
    depth = np.random.randn(80, 100).astype(np.float32)
    state = np.random.randn(8).astype(np.float32)
    action = agent.select_action(depth, state)
    print(f"✓ Action shape: {action.shape}")
    print(f"✓ Action values: {action}")
    
    print("\n✓ TD3 implementation works!")

