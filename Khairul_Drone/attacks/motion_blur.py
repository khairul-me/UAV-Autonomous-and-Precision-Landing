from __future__ import annotations

import cv2
import numpy as np

from attacks.base_attack import AdversarialAttack


class MotionBlur(AdversarialAttack):
    """Simulate camera motion blur."""

    def __init__(self, kernel_size: int = 15, angle: float = 45.0) -> None:
        super().__init__(model=None, device="cpu")
        self.kernel_size = kernel_size
        self.angle = angle

    def attack(
        self, depth_image: np.ndarray, state: np.ndarray, target: np.ndarray | None = None
    ) -> np.ndarray:
        kernel = self._motion_blur_kernel(self.kernel_size, self.angle)
        blurred = cv2.filter2D(depth_image, -1, kernel)
        return blurred

    def _motion_blur_kernel(self, size: int, angle: float) -> np.ndarray:
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1.0
        kernel /= size

        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, matrix, (size, size))
        kernel = kernel / (kernel.sum() + 1e-8)
        return kernel

