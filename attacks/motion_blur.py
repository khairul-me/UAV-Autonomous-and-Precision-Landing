# attacks/motion_blur.py
import cv2
import numpy as np
from attacks import AdversarialAttack

class MotionBlurAttack(AdversarialAttack):
    """
    Motion Blur Attack
    
    Insight from E-DQN paper: Drones are vulnerable to motion blur!
    This exploits that vulnerability
    
    Physical attack: Can happen naturally or be induced
    """
    
    def __init__(self, kernel_size=11, angle=0):
        """
        Args:
            kernel_size: Size of motion blur kernel (larger = more blur)
            angle: Direction of motion blur (degrees)
        """
        super().__init__(name="MotionBlur")
        self.kernel_size = kernel_size
        self.angle = angle
    
    def attack(self, depth_image, **kwargs):
        """Apply motion blur to depth image"""
        
        # Create motion blur kernel
        kernel = self._get_motion_kernel(self.kernel_size, self.angle)
        
        # Apply blur
        blurred = cv2.filter2D(depth_image, -1, kernel)
        
        return blurred
    
    def _get_motion_kernel(self, size, angle):
        """Generate motion blur kernel"""
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        
        # Rotate kernel
        M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        
        return kernel

# Test
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Create test depth image
    clean_depth = np.random.rand(80, 100).astype(np.float32)
    
    # Test different blur strengths
    blur_mild = MotionBlurAttack(kernel_size=5)
    blur_moderate = MotionBlurAttack(kernel_size=11)
    blur_strong = MotionBlurAttack(kernel_size=21)
    
    depth_mild = blur_mild.attack(clean_depth)
    depth_moderate = blur_moderate.attack(clean_depth)
    depth_strong = blur_strong.attack(clean_depth)
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(clean_depth, cmap='gray')
    axes[0].set_title('Clean')
    
    axes[1].imshow(depth_mild, cmap='gray')
    axes[1].set_title('Mild Blur (k=5)')
    
    axes[2].imshow(depth_moderate, cmap='gray')
    axes[2].set_title('Moderate Blur (k=11)')
    
    axes[3].imshow(depth_strong, cmap='gray')
    axes[3].set_title('Strong Blur (k=21)')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('motion_blur_attack.png', dpi=150)
    print("[OK] Motion blur attack works!")

