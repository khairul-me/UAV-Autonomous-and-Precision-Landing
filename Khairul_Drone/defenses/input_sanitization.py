from __future__ import annotations

import cv2
import numpy as np


class InputSanitizer:
    """Apply simple defenses to mitigate adversarial depth perturbations."""

    def sanitize(self, depth_image: np.ndarray) -> np.ndarray:
        cleaned = cv2.medianBlur(depth_image.astype(np.float32), 5)
        cleaned = self._tv_denoise(cleaned)
        cleaned = self._jpeg_compression(cleaned)
        return cleaned

    def _tv_denoise(
        self, image: np.ndarray, weight: float = 0.1, iterations: int = 5
    ) -> np.ndarray:
        result = image.copy()
        for _ in range(iterations):
            grad_x = np.roll(result, 1, axis=1) - result
            grad_y = np.roll(result, 1, axis=0) - result
            magnitude = np.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            result = result + weight * (grad_x / magnitude + grad_y / magnitude)
        return result

    def _jpeg_compression(self, image: np.ndarray, quality: int = 75) -> np.ndarray:
        max_val = float(np.max(image)) if np.max(image) > 0 else 1.0
        image_uint8 = np.clip((image / max_val) * 255.0, 0, 255).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded = cv2.imencode(".jpg", image_uint8, encode_param)
        if not success:
            return image
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        return (decoded.astype(np.float32) / 255.0) * max_val

