# models/feature_extractor.py
import torch
import torch.nn as nn

class DepthFeatureExtractor(nn.Module):
    """
    CNN for extracting features from depth images
    Based on DPRL paper's architecture (Table 1)
    """
    
    def __init__(self, input_channels=1, output_dim=25):
        super(DepthFeatureExtractor, self).__init__()
        
        # Conv Block 1: 1×80×100 → 8×80×100 → 8×40×50
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 2: 8×40×50 → 16×40×50 → 16×20×25
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 3: 16×20×25 → 25×20×25 → 25×10×12
        self.conv3 = nn.Conv2d(16, 25, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(25)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling: 25×10×12 → 25×1×1 → 25
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: Depth image tensor [batch, 1, 80, 100]
        Returns:
            features: Feature vector [batch, 25]
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)  # [batch, 8, 40, 50]
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)  # [batch, 16, 20, 25]
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)  # [batch, 25, 10, 12]
        
        # Global pooling
        x = self.global_pool(x)  # [batch, 25, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 25]
        
        return x

# Test the feature extractor
if __name__ == "__main__":
    extractor = DepthFeatureExtractor()
    
    # Test with dummy depth image
    dummy_depth = torch.randn(4, 1, 80, 100)  # Batch of 4 images
    features = extractor(dummy_depth)
    
    print(f"Input shape: {dummy_depth.shape}")
    print(f"Output shape: {features.shape}")
    print(f"[OK] Feature extractor works! Output: [batch_size, 25]")

