# Attacks package
import torch
import numpy as np

class AdversarialAttack:
    """
    Base class for adversarial attacks on depth images
    """
    
    def __init__(self, name="BaseAttack"):
        self.name = name
    
    def attack(self, depth_image, model=None, target=None):
        """
        Generate adversarial perturbation
        
        Args:
            depth_image: numpy array [H, W] or torch tensor
            model: Target model (for gradient-based attacks)
            target: Target output (for targeted attacks)
        
        Returns:
            Adversarial depth image (same format as input)
        """
        raise NotImplementedError
    
    def __call__(self, depth_image, model=None, target=None):
        return self.attack(depth_image, model, target)

