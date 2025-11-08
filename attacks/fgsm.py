# attacks/fgsm.py
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks import AdversarialAttack

class FGSM(AdversarialAttack):
    """
    Fast Gradient Sign Method (FGSM)
    
    Simple but effective attack that adds noise in direction of gradient:
    x_adv = x + epsilon * sign(∇_x Loss)
    
    This is your FIRST attack - start here!
    """
    
    def __init__(self, epsilon=0.03, clip_min=0.0, clip_max=1.0):
        """
        Args:
            epsilon: Perturbation magnitude (0.03 = 3% of depth range)
            clip_min: Minimum depth value
            clip_max: Maximum depth value
        """
        super().__init__(name="FGSM")
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def attack(self, depth_image, actor_model, state, target_action=None):
        """
        Generate FGSM adversarial depth image
        
        Args:
            depth_image: numpy array [80, 100]
            actor_model: Actor network
            state: Self-state vector [8]
            target_action: If None, untargeted attack (maximize loss)
        
        Returns:
            Adversarial depth image [80, 100]
        """
        # Convert to torch tensor
        device = next(actor_model.parameters()).device
        
        depth_tensor = torch.FloatTensor(depth_image).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 80, 100]
        depth_tensor.requires_grad = True
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # [1, 8]
        
        # Forward pass
        action = actor_model(depth_tensor, state_tensor)
        
        # Compute loss
        if target_action is None:
            # Untargeted: maximize action variance (cause unpredictable behavior)
            loss = -torch.var(action)
        else:
            # Targeted: push action towards target
            target_tensor = torch.FloatTensor(target_action).unsqueeze(0).to(device)
            loss = nn.MSELoss()(action, target_tensor)
        
        # Backward pass
        actor_model.zero_grad()
        loss.backward()
        
        # Get gradient
        gradient = depth_tensor.grad.data
        
        # Generate adversarial perturbation: epsilon * sign(gradient)
        perturbation = self.epsilon * torch.sign(gradient)
        
        # Apply perturbation
        adv_depth_tensor = depth_tensor + perturbation
        
        # Clip to valid range
        adv_depth_tensor = torch.clamp(adv_depth_tensor, self.clip_min, self.clip_max)
        
        # Convert back to numpy
        adv_depth = adv_depth_tensor.squeeze().cpu().detach().numpy()
        
        return adv_depth

# Test FGSM
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.actor import Actor
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Create dummy actor model
    actor = Actor()
    actor.eval()
    
    # Create clean depth image
    clean_depth = np.random.rand(80, 100).astype(np.float32)
    state = np.random.randn(8).astype(np.float32)
    
    # Create FGSM attack
    attack = FGSM(epsilon=0.05)
    
    # Generate adversarial example
    adv_depth = attack.attack(clean_depth, actor, state)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(clean_depth, cmap='gray')
    axes[0].set_title('Clean Depth')
    axes[0].axis('off')
    
    axes[1].imshow(adv_depth, cmap='gray')
    axes[1].set_title('Adversarial Depth (FGSM)')
    axes[1].axis('off')
    
    perturbation = adv_depth - clean_depth
    axes[2].imshow(perturbation, cmap='RdBu', vmin=-0.1, vmax=0.1)
    axes[2].set_title('Perturbation (amplified)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('fgsm_attack_example.png', dpi=150)
    print("[OK] FGSM attack works!")
    print(f"  Perturbation L-infinity norm: {np.abs(perturbation).max():.4f}")
    print(f"  Perturbation L2 norm: {np.linalg.norm(perturbation):.4f}")

