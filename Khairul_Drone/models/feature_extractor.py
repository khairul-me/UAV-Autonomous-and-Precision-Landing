import torch
import torch.nn as nn


class DepthFeatureExtractor(nn.Module):
    """Simple CNN that outputs a 25D feature vector from depth images."""

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(16, 25, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(depth))
        x = self.pool1(x)

        x = self.activation(self.conv2(x))
        x = self.pool2(x)

        x = self.activation(self.conv3(x))
        x = self.pool3(x)

        return x.view(x.size(0), -1)

