"""
Multi-camera capture and processing utilities for the enhanced AirSim setup.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import airsim
import cv2
import numpy as np
import torch
import torch.nn as nn


class MultiCameraManager:
    """
    Handles synchronous capture across all configured cameras and provides
    helper utilities (obstacle distance estimation, composite visualisation).
    """

    def __init__(self, client: airsim.MultirotorClient):
        self.client = client

        # Reference configuration matches settings_enhanced.json
        self.cameras = {
            "front_center": {
                "resolution": (320, 240),
                "fov": 90,
                "types": ["rgb", "depth", "segmentation"],
            },
            "front_left": {
                "resolution": (160, 120),
                "fov": 90,
                "types": ["depth"],
            },
            "front_right": {
                "resolution": (160, 120),
                "fov": 90,
                "types": ["depth"],
            },
            "bottom": {
                "resolution": (256, 256),
                "fov": 90,
                "types": ["rgb", "depth"],
            },
        }

        self.image_type_map = {
            "rgb": airsim.ImageType.Scene,
            "depth": airsim.ImageType.DepthPlanar,
            "segmentation": airsim.ImageType.Segmentation,
        }

        print(f"[OK] MultiCameraManager initialised with {len(self.cameras)} cameras")

    # --------------------------------------------------------------------- #
    # Capture helpers
    # --------------------------------------------------------------------- #
    def capture_all(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Capture every configured stream in a single RPC call.

        Returns
        -------
        Dict mapping camera -> {stream -> ndarray}.
        Only streams with valid data are returned.
        """
        requests: List[airsim.ImageRequest] = []
        request_index: List[Tuple[str, str]] = []

        for cam_name, conf in self.cameras.items():
            for stream in conf["types"]:
                requests.append(
                    airsim.ImageRequest(
                        cam_name,
                        self.image_type_map[stream],
                        pixels_as_float=(stream == "depth"),
                        compress=False,
                    )
                )
                request_index.append((cam_name, stream))

        responses = self.client.simGetImages(requests)
        result: Dict[str, Dict[str, np.ndarray]] = {cam: {} for cam in self.cameras}

        for response, (cam_name, stream) in zip(responses, request_index):
            if response.height <= 0 or response.width <= 0:
                continue
            result[cam_name][stream] = self._parse_image(response, stream)

        return result

    def get_depth_images(self) -> Dict[str, np.ndarray]:
        """Return depth frames for every camera that supplies depth."""
        requests: List[airsim.ImageRequest] = []
        cam_order: List[str] = []
        for cam_name, conf in self.cameras.items():
            if "depth" in conf["types"]:
                requests.append(
                    airsim.ImageRequest(cam_name, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)
                )
                cam_order.append(cam_name)

        responses = self.client.simGetImages(requests)
        result: Dict[str, np.ndarray] = {}
        for response, cam_name in zip(responses, cam_order):
            if response.height <= 0 or response.width <= 0:
                continue
            depth = np.array(response.image_data_float, dtype=np.float32).reshape(response.height, response.width)
            result[cam_name] = depth
        return result

    # --------------------------------------------------------------------- #
    # Processing helpers
    # --------------------------------------------------------------------- #
    def compute_obstacle_distances(self, depth_images: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Estimate obstacle distance per camera by sampling the lowest 5th percentile depth.
        """
        distances: Dict[str, float] = {}
        for cam_name, depth in depth_images.items():
            valid = depth[np.isfinite(depth)]
            valid = valid[valid < 100.0]  # Cap to sensor range
            if valid.size == 0:
                distances[cam_name] = 100.0
            else:
                distances[cam_name] = float(np.percentile(valid, 5))
        return distances

    def visualize_multi_view(self, images: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Build a composite diagnostic view showing all camera feeds.
        """
        height, width = 240, 320
        composite = np.zeros((height * 2, width * 3, 3), dtype=np.uint8)

        def place(image: np.ndarray, top: int, left: int):
            h, w = image.shape[:2]
            composite[top : top + h, left : left + w] = image

        # Top row
        if "front_left" in images and "depth" in images["front_left"]:
            depth_vis = self._depth_to_rgb(images["front_left"]["depth"])
            place(cv2.resize(depth_vis, (width, height)), 0, 0)

        if "front_center" in images and "rgb" in images["front_center"]:
            rgb = cv2.cvtColor(images["front_center"]["rgb"], cv2.COLOR_RGB2BGR)
            place(cv2.resize(rgb, (width, height)), 0, width)

        if "front_right" in images and "depth" in images["front_right"]:
            depth_vis = self._depth_to_rgb(images["front_right"]["depth"])
            place(cv2.resize(depth_vis, (width, height)), 0, width * 2)

        # Bottom row
        if "front_center" in images and "depth" in images["front_center"]:
            depth_vis = self._depth_to_rgb(images["front_center"]["depth"])
            place(cv2.resize(depth_vis, (width, height)), height, 0)

        if "front_center" in images and "segmentation" in images["front_center"]:
            seg = cv2.cvtColor(images["front_center"]["segmentation"], cv2.COLOR_RGB2BGR)
            place(cv2.resize(seg, (width, height)), height, width)

        if "bottom" in images and "rgb" in images["bottom"]:
            bottom_rgb = cv2.cvtColor(images["bottom"]["rgb"], cv2.COLOR_RGB2BGR)
            place(cv2.resize(bottom_rgb, (width, height)), height, width * 2)

        # Labelling
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(composite, "LEFT DEPTH", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(composite, "FRONT RGB", (width + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(composite, "RIGHT DEPTH", (width * 2 + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(composite, "FRONT DEPTH", (10, height + 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(composite, "FRONT SEG", (width + 10, height + 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(composite, "BOTTOM RGB", (width * 2 + 10, height + 30), font, 0.7, (255, 255, 255), 2)

        return composite

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _parse_image(self, response: airsim.ImageResponse, stream: str) -> np.ndarray:
        if stream == "rgb" or stream == "segmentation":
            img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            return img.reshape(response.height, response.width, 3)
        if stream == "depth":
            img = np.array(response.image_data_float, dtype=np.float32)
            return img.reshape(response.height, response.width)
        raise ValueError(f"Unsupported stream type: {stream}")

    def _depth_to_rgb(self, depth: np.ndarray, max_depth: float = 50.0) -> np.ndarray:
        depth_norm = np.clip(depth, 0, max_depth) / max_depth
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        try:
            colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
        except AttributeError:
            # Fallback for older OpenCV builds
            colormap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        return colormap


class MultiCameraFeatureExtractor(nn.Module):
    """
    CNN feature extractor that fuses depth views from all cameras into a 32D embedding.
    """

    def __init__(self):
        super().__init__()

        self.front_center_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.side_extractor = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.bottom_extractor = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(32 + 16 + 16 + 16, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
        )

        print("[OK] MultiCameraFeatureExtractor initialised (output=32D)")

    def forward(self, depth_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []

        if "front_center" in depth_dict:
            depth = depth_dict["front_center"].unsqueeze(1)
            features.append(self.front_center_extractor(depth))

        if "front_left" in depth_dict:
            depth = depth_dict["front_left"].unsqueeze(1)
            features.append(self.side_extractor(depth))

        if "front_right" in depth_dict:
            depth = depth_dict["front_right"].unsqueeze(1)
            features.append(self.side_extractor(depth))

        if "bottom" in depth_dict:
            depth = depth_dict["bottom"].unsqueeze(1)
            features.append(self.bottom_extractor(depth))

        if not features:
            raise ValueError("No depth tensors provided to feature extractor")

        combined = torch.cat(features, dim=1)
        return self.fusion(combined)


# ------------------------------------------------------------------------- #
# Diagnostic script
# ------------------------------------------------------------------------- #
def test_multi_camera():
    """Manual test entry point."""
    import matplotlib.pyplot as plt

    print("Testing MultiCameraManager...")
    client = airsim.MultirotorClient()
    client.confirmConnection()

    manager = MultiCameraManager(client)

    print("\nStep 1: capturing all streams")
    images = manager.capture_all()
    print(f"[OK] Streams captured: { {k: list(v.keys()) for k, v in images.items()} }")

    print("\nStep 2: fetching depth images only")
    depth_images = manager.get_depth_images()
    print(f"[OK] Depth cameras: {list(depth_images.keys())}")

    print("\nStep 3: obstacle distance estimation")
    distances = manager.compute_obstacle_distances(depth_images)
    for cam, dist in distances.items():
        print(f"  {cam}: {dist:.2f} m")

    print("\nStep 4: composite visualisation -> outputs/multi_camera_test.png")
    composite = manager.visualize_multi_view(images)
    Path("outputs").mkdir(exist_ok=True)
    cv2.imwrite("outputs/multi_camera_test.png", composite)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Multi-camera view")
    plt.tight_layout()
    plt.savefig("outputs/multi_camera_test_preview.png")
    plt.close()

    print("\nStep 5: feature extraction")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = MultiCameraFeatureExtractor().to(device)

    depth_tensors: Dict[str, torch.Tensor] = {}
    for cam_name, depth in depth_images.items():
        # Resize depth to maintain manageable tensor shapes for small conv nets.
        if cam_name == "front_center":
            resized = cv2.resize(depth, (240, 120))
        elif cam_name in ("front_left", "front_right"):
            resized = cv2.resize(depth, (160, 80))
        elif cam_name == "bottom":
            resized = cv2.resize(depth, (160, 160))
        else:
            resized = depth
        tensor = torch.from_numpy(resized).float().unsqueeze(0).to(device)
        depth_tensors[cam_name] = tensor

    features = extractor(depth_tensors)
    print(f"[OK] Feature tensor shape: {tuple(features.shape)}")
    print(f"[OK] Feature range: [{features.min().item():.3f}, {features.max().item():.3f}]")

    print("\n[OK] Multi-camera utilities test complete.")


if __name__ == "__main__":
    test_multi_camera()

