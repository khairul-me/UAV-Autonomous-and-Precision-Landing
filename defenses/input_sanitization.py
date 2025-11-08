# defenses/input_sanitization.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DenoisingAutoencoder(nn.Module):
    """
    Autoencoder to remove adversarial perturbations from depth images
    Inspired by E-DQN paper's event processing autoencoder
    
    Learns to map noisy/attacked images back to clean versions
    """
    
    def __init__(self, input_channels=1):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder: Compress image while removing noise
        self.encoder = nn.Sequential(
            # 1×80×100 → 16×40×50
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # 16×40×50 → 32×20×25
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 32×20×25 → 64×10×12
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 64×10×12 → 128×5×6
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Decoder: Reconstruct clean image
        self.decoder = nn.Sequential(
            # 128×5×6 → 64×10×12
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 64×10×12 → 32×20×25
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 32×20×25 → 16×40×50
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # 16×40×50 → 1×80×100
            nn.ConvTranspose2d(16, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x):
        """
        Args:
            x: Noisy/attacked depth image [batch, 1, 80, 100]
        Returns:
            Denoised depth image [batch, 1, 80, 100]
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def denoise(self, depth_image):
        """
        Denoise a single depth image
        
        Args:
            depth_image: numpy array [80, 100] or torch tensor
        Returns:
            Denoised depth image (same format as input)
        """
        was_numpy = isinstance(depth_image, np.ndarray)
        device = next(self.parameters()).device
        
        # Convert to torch tensor
        if was_numpy:
            depth_tensor = torch.FloatTensor(depth_image).unsqueeze(0).unsqueeze(0).to(device)
        else:
            depth_tensor = depth_image
            if len(depth_tensor.shape) == 2:
                depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
        
        # Denoise
        self.eval()
        with torch.no_grad():
            denoised = self.forward(depth_tensor)
        
        # Convert back
        if was_numpy:
            return denoised.squeeze().cpu().numpy()
        else:
            return denoised


class AnomalyDetector(nn.Module):
    """
    Detect anomalies in depth images that might indicate adversarial attacks
    Uses statistical properties of depth images
    """
    
    def __init__(self, threshold=0.1):
        super(AnomalyDetector, self).__init__()
        self.threshold = threshold
        
        # Track statistics of clean images
        self.register_buffer('mean_histogram', torch.zeros(20))
        self.register_buffer('std_histogram', torch.ones(20))
        self.calibrated = False
    
    def calibrate(self, clean_images, num_bins=20):
        """
        Calibrate detector using clean images
        
        Args:
            clean_images: List of clean depth images [N, H, W]
        """
        histograms = []
        
        for img in clean_images:
            if isinstance(img, np.ndarray):
                img = torch.FloatTensor(img)
            
            # Compute histogram
            hist = torch.histc(img.flatten(), bins=num_bins, min=0, max=1)
            hist = hist / hist.sum()  # Normalize
            histograms.append(hist)
        
        histograms = torch.stack(histograms)
        
        # Compute mean and std of histogram distributions
        self.mean_histogram = histograms.mean(dim=0)
        self.std_histogram = histograms.std(dim=0) + 1e-6
        
        self.calibrated = True
        print(f"[OK] Anomaly detector calibrated on {len(clean_images)} images")
    
    def detect(self, depth_image, return_score=False):
        """
        Detect if depth image is anomalous (possibly attacked)
        
        Args:
            depth_image: numpy array [H, W] or torch tensor
            return_score: If True, return anomaly score
        
        Returns:
            is_anomaly (bool) or (is_anomaly, anomaly_score)
        """
        if not self.calibrated:
            raise RuntimeError("Detector must be calibrated before use!")
        
        if isinstance(depth_image, np.ndarray):
            img = torch.FloatTensor(depth_image)
        else:
            img = depth_image
        
        # Compute histogram
        hist = torch.histc(img.flatten(), bins=20, min=0, max=1)
        hist = hist / (hist.sum() + 1e-6)
        
        # Compute z-score distance from normal distribution
        z_scores = (hist - self.mean_histogram) / self.std_histogram
        anomaly_score = torch.abs(z_scores).max().item()
        
        is_anomaly = anomaly_score > self.threshold
        
        if return_score:
            return is_anomaly, anomaly_score
        else:
            return is_anomaly


class InputSanitizer:
    """
    Complete input sanitization pipeline
    Combines denoising + anomaly detection
    """
    
    def __init__(self, denoiser, anomaly_detector):
        self.denoiser = denoiser
        self.anomaly_detector = anomaly_detector
    
    def sanitize(self, depth_image, detect_only=False):
        """
        Sanitize input depth image
        
        Args:
            depth_image: Input depth image (possibly attacked)
            detect_only: If True, only detect without denoising
        
        Returns:
            {
                'clean_image': Sanitized image,
                'is_anomaly': Whether attack was detected,
                'anomaly_score': Anomaly confidence score
            }
        """
        # Detect anomaly
        is_anomaly, score = self.anomaly_detector.detect(depth_image, return_score=True)
        
        if detect_only:
            return {
                'clean_image': depth_image,
                'is_anomaly': is_anomaly,
                'anomaly_score': score
            }
        
        # Denoise if anomaly detected
        if is_anomaly:
            clean_image = self.denoiser.denoise(depth_image)
        else:
            clean_image = depth_image
        
        return {
            'clean_image': clean_image,
            'is_anomaly': is_anomaly,
            'anomaly_score': score
        }


# Training script for denoising autoencoder
def train_denoising_autoencoder(clean_images, attacked_images, epochs=50, batch_size=32, lr=1e-3):
    """
    Train denoising autoencoder
    
    Args:
        clean_images: List of clean depth images
        attacked_images: List of corresponding attacked images
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    
    Returns:
        Trained denoiser
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    denoiser = DenoisingAutoencoder().to(device)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Prepare data
    clean_tensors = torch.stack([torch.FloatTensor(img) for img in clean_images])
    attacked_tensors = torch.stack([torch.FloatTensor(img) for img in attacked_images])
    
    dataset = torch.utils.data.TensorDataset(attacked_tensors, clean_tensors)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("Training Denoising Autoencoder...")
    print("="*60)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for noisy_batch, clean_batch in loader:
            noisy_batch = noisy_batch.unsqueeze(1).to(device)  # [B, 1, 80, 100]
            clean_batch = clean_batch.unsqueeze(1).to(device)
            
            # Forward pass
            reconstructed = denoiser(noisy_batch)
            loss = criterion(reconstructed, clean_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    print("="*60)
    print("[OK] Training complete!")
    
    return denoiser


# Test
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    try:
        from attacks.fgsm import FGSM
        from models.actor import Actor
    except ImportError:
        print("Warning: Could not import attacks/models. Using dummy data.")
        FGSM = None
        Actor = None
    
    print("Testing Input Sanitization Defense...")
    print("="*60)
    
    # Create dummy data
    print("\n1. Generating test data...")
    clean_images = [np.random.rand(80, 100).astype(np.float32) for _ in range(100)]
    
    # Create attacks
    if FGSM is not None and Actor is not None:
        actor = Actor()
        actor.eval()
        fgsm = FGSM(epsilon=0.05)
        
        attacked_images = []
        for clean in clean_images:
            state = np.random.randn(8).astype(np.float32)
            attacked = fgsm.attack(clean, actor, state)
            attacked_images.append(attacked)
    else:
        # Dummy attacked images
        attacked_images = [img + np.random.randn(80, 100).astype(np.float32) * 0.05 for img in clean_images]
        attacked_images = [np.clip(img, 0, 1) for img in attacked_images]
    
    print(f"[OK] Generated {len(clean_images)} clean and attacked images")
    
    # Train denoiser
    print("\n2. Training denoiser...")
    denoiser = train_denoising_autoencoder(
        clean_images[:80], 
        attacked_images[:80],
        epochs=30,
        batch_size=16
    )
    
    # Calibrate anomaly detector
    print("\n3. Calibrating anomaly detector...")
    detector = AnomalyDetector(threshold=0.15)
    detector.calibrate(clean_images[:80])
    
    # Create sanitizer
    sanitizer = InputSanitizer(denoiser, detector)
    
    # Test on held-out data
    print("\n4. Testing on held-out data...")
    test_clean = clean_images[80:90]
    test_attacked = attacked_images[80:90]
    
    clean_detected = 0
    attacked_detected = 0
    
    for img in test_clean:
        result = sanitizer.sanitize(img, detect_only=True)
        if result['is_anomaly']:
            clean_detected += 1
    
    for img in test_attacked:
        result = sanitizer.sanitize(img, detect_only=True)
        if result['is_anomaly']:
            attacked_detected += 1
    
    print(f"\nDetection Results:")
    print(f"  False positive rate: {clean_detected/len(test_clean)*100:.1f}%")
    print(f"  True positive rate: {attacked_detected/len(test_attacked)*100:.1f}%")
    
    # Visualize denoising
    print("\n5. Visualizing results...")
    test_idx = 0
    clean = test_clean[test_idx]
    attacked = test_attacked[test_idx]
    result = sanitizer.sanitize(attacked)
    denoised = result['clean_image']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(clean, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Clean Original')
    axes[0].axis('off')
    
    axes[1].imshow(attacked, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Attacked\n(Anomaly: {result["is_anomaly"]})')
    axes[1].axis('off')
    
    axes[2].imshow(denoised, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Denoised')
    axes[2].axis('off')
    
    axes[3].imshow(np.abs(clean - denoised), cmap='hot', vmin=0, vmax=0.2)
    axes[3].set_title('Reconstruction Error')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('input_sanitization_test.png', dpi=150)
    print("[OK] Saved visualization to input_sanitization_test.png")
    
    print("\n" + "="*60)
    print("[OK] Input Sanitization Defense Test Complete!")

