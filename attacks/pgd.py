# attacks/pgd.py
import torch
import torch.nn as nn
import numpy as np
from attacks import AdversarialAttack

class PGD(AdversarialAttack):
    """
    Projected Gradient Descent (PGD)
    
    Iterative version of FGSM - much more powerful!
    Applies FGSM multiple times with smaller steps
    
    This is a STRONG attack for evaluation
    """
    
    def __init__(self, epsilon=0.03, alpha=0.007, num_steps=10, 
                 clip_min=0.0, clip_max=1.0, random_start=True):
        """
        Args:
            epsilon: Max perturbation (L∞ bound)
            alpha: Step size for each iteration
            num_steps: Number of iterations
            random_start: Start from random point in epsilon ball
        """
        super().__init__(name="PGD")
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.random_start = random_start
    
    def attack(self, depth_image, actor_model, state, target_action=None):
        """Generate PGD adversarial example"""
        
        device = next(actor_model.parameters()).device
        
        # Original depth image
        orig_depth = torch.FloatTensor(depth_image).to(device)
        
        # Initialize adversarial depth
        if self.random_start:
            # Start from random point in epsilon ball
            adv_depth = orig_depth + torch.empty_like(orig_depth).uniform_(
                -self.epsilon, self.epsilon
            )
            adv_depth = torch.clamp(adv_depth, self.clip_min, self.clip_max)
        else:
            adv_depth = orig_depth.clone()
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Iterative attack
        for i in range(self.num_steps):
            adv_depth_batch = adv_depth.unsqueeze(0).unsqueeze(0)  # [1, 1, 80, 100]
            adv_depth_batch.requires_grad = True
            
            # Forward pass
            action = actor_model(adv_depth_batch, state_tensor)
            
            # Compute loss
            if target_action is None:
                # Untargeted: cause bad decisions
                loss = -torch.var(action)
            else:
                target_tensor = torch.FloatTensor(target_action).unsqueeze(0).to(device)
                loss = nn.MSELoss()(action, target_tensor)
            
            # Backward
            actor_model.zero_grad()
            loss.backward()
            
            # Get gradient
            gradient = adv_depth_batch.grad.data.squeeze()
            
            # Take step in direction of gradient
            adv_depth = adv_depth + self.alpha * torch.sign(gradient)
            
            # Project back to epsilon ball around original image
            perturbation = torch.clamp(adv_depth - orig_depth, -self.epsilon, self.epsilon)
            adv_depth = orig_depth + perturbation
            
            # Clip to valid range
            adv_depth = torch.clamp(adv_depth, self.clip_min, self.clip_max)
        
        return adv_depth.cpu().detach().numpy()

# Test PGD
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.actor import Actor
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    actor = Actor()
    actor.eval()
    
    clean_depth = np.random.rand(80, 100).astype(np.float32)
    state = np.random.randn(8).astype(np.float32)
    
    # Compare FGSM vs PGD
    from attacks.fgsm import FGSM
    
    fgsm_attack = FGSM(epsilon=0.03)
    pgd_attack = PGD(epsilon=0.03, alpha=0.007, num_steps=10)
    
    fgsm_depth = fgsm_attack.attack(clean_depth, actor, state)
    pgd_depth = pgd_attack.attack(clean_depth, actor, state)
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(clean_depth, cmap='gray')
    axes[0].set_title('Clean Depth')
    
    axes[1].imshow(fgsm_depth, cmap='gray')
    axes[1].set_title('FGSM Attack')
    
    axes[2].imshow(pgd_depth, cmap='gray')
    axes[2].set_title('PGD Attack (Stronger)')
    
    axes[3].imshow(pgd_depth - fgsm_depth, cmap='RdBu')
    axes[3].set_title('PGD vs FGSM Difference')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('pgd_attack_example.png', dpi=150)
    print("[OK] PGD attack works!")

