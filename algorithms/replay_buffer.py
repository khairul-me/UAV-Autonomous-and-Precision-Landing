# algorithms/replay_buffer.py
import numpy as np
import torch

class ReplayBuffer:
    """
    Experience Replay Buffer for off-policy RL
    Stores transitions: (s, a, r, s', done)
    """
    
    def __init__(self, max_size=50000, depth_shape=(80, 100), state_dim=8, action_dim=4):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # Preallocate memory for efficiency
        self.depth = np.zeros((max_size, *depth_shape), dtype=np.float32)
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_depth = np.zeros((max_size, *depth_shape), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        
        # For privileged learning: store clean depth images
        self.clean_depth = np.zeros((max_size, *depth_shape), dtype=np.float32)
        self.next_clean_depth = np.zeros((max_size, *depth_shape), dtype=np.float32)
    
    def add(self, depth, state, action, reward, next_depth, next_state, done,
            clean_depth=None, next_clean_depth=None):
        """Add transition to buffer"""
        
        self.depth[self.ptr] = depth
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_depth[self.ptr] = next_depth
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        
        # Store clean versions for privileged learning
        if clean_depth is not None:
            self.clean_depth[self.ptr] = clean_depth
        else:
            self.clean_depth[self.ptr] = depth  # If no attack, same as noisy
            
        if next_clean_depth is not None:
            self.next_clean_depth[self.ptr] = next_clean_depth
        else:
            self.next_clean_depth[self.ptr] = next_depth
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size=128, use_privileged=False):
        """
        Sample batch from buffer
        
        Args:
            batch_size: Number of transitions to sample
            use_privileged: If True, return clean depth for Critic (privileged learning)
        
        Returns:
            Dictionary with sampled transitions
        """
        ind = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            'depth': torch.FloatTensor(self.depth[ind]),
            'state': torch.FloatTensor(self.state[ind]),
            'action': torch.FloatTensor(self.action[ind]),
            'reward': torch.FloatTensor(self.reward[ind]),
            'next_depth': torch.FloatTensor(self.next_depth[ind]),
            'next_state': torch.FloatTensor(self.next_state[ind]),
            'done': torch.FloatTensor(self.done[ind])
        }
        
        # For privileged learning: return clean depth for Critic
        if use_privileged:
            batch['clean_depth'] = torch.FloatTensor(self.clean_depth[ind])
            batch['next_clean_depth'] = torch.FloatTensor(self.next_clean_depth[ind])
        
        return batch
    
    def __len__(self):
        return self.size

# Test
if __name__ == "__main__":
    buffer = ReplayBuffer(max_size=1000)
    
    # Add dummy transitions
    for i in range(100):
        depth = np.random.randn(80, 100).astype(np.float32)
        state = np.random.randn(8).astype(np.float32)
        action = np.random.randn(4).astype(np.float32)
        reward = np.random.randn()
        next_depth = np.random.randn(80, 100).astype(np.float32)
        next_state = np.random.randn(8).astype(np.float32)
        done = 0.0
        
        buffer.add(depth, state, action, reward, next_depth, next_state, done)
    
    # Sample batch
    batch = buffer.sample(batch_size=32)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Sampled depth shape: {batch['depth'].shape}")
    print(f"Sampled state shape: {batch['state'].shape}")
    print(f"[OK] Replay buffer works!")

