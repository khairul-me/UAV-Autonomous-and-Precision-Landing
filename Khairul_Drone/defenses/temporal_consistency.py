from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import numpy as np


class TemporalConsistencyChecker:
    """Detect sudden temporal shifts indicative of adversarial influence."""

    def __init__(self, window_size: int = 10) -> None:
        self.window_size = window_size
        self.depth_history: Deque[np.ndarray] = deque(maxlen=window_size)
        self.action_history: Deque[np.ndarray] = deque(maxlen=window_size)

    def check_anomaly(
        self, depth_image: np.ndarray, action: np.ndarray
    ) -> Tuple[bool, str]:
        self.depth_history.append(depth_image.copy())
        self.action_history.append(action.copy())

        if len(self.depth_history) < self.window_size:
            return False, "Insufficient history"

        depth_changes = [
            np.linalg.norm(self.depth_history[i] - self.depth_history[i - 1])
            for i in range(1, len(self.depth_history))
        ]
        if depth_changes and depth_changes[-1] > 2.0 * np.mean(depth_changes[:-1] or [1.0]):
            return True, "Depth change anomaly"

        action_changes = [
            np.linalg.norm(self.action_history[i] - self.action_history[i - 1])
            for i in range(1, len(self.action_history))
        ]
        if action_changes and action_changes[-1] > 2.0 * np.mean(action_changes[:-1] or [1.0]):
            return True, "Action change anomaly"

        return False, "Temporal pattern nominal"

