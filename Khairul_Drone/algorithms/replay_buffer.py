from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List

import numpy as np
import torch


class ReplayBuffer:
    """Simple replay buffer for off-policy algorithms."""

    def __init__(self, max_size: int = 50_000) -> None:
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.depth: List[np.ndarray] = []
        self.state: List[np.ndarray] = []
        self.action: List[np.ndarray] = []
        self.reward: List[float] = []
        self.next_depth: List[np.ndarray] = []
        self.next_state: List[np.ndarray] = []
        self.done: List[bool] = []

    def add(
        self,
        depth: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_depth: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if self.size < self.max_size:
            self.depth.append(depth)
            self.state.append(state)
            self.action.append(action)
            self.reward.append(reward)
            self.next_depth.append(next_depth)
            self.next_state.append(next_state)
            self.done.append(done)
            self.size += 1
        else:
            self.depth[self.ptr] = depth
            self.state[self.ptr] = state
            self.action[self.ptr] = action
            self.reward[self.ptr] = reward
            self.next_depth[self.ptr] = next_depth
            self.next_state[self.ptr] = next_state
            self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size: int = 128) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)

        batch = {
            "depth": torch.FloatTensor([self.depth[i] for i in indices]),
            "state": torch.FloatTensor([self.state[i] for i in indices]),
            "action": torch.FloatTensor([self.action[i] for i in indices]),
            "reward": torch.FloatTensor([self.reward[i] for i in indices]).unsqueeze(1),
            "next_depth": torch.FloatTensor([self.next_depth[i] for i in indices]),
            "next_state": torch.FloatTensor([self.next_state[i] for i in indices]),
            "done": torch.FloatTensor([self.done[i] for i in indices]).unsqueeze(1),
        }

        return batch

    def __len__(self) -> int:
        return self.size

