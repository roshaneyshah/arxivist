"""Simple FIFO replay buffer (capacity 20 000, per Table 4)."""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.buffer: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, s, a, r, s_next, done) -> None:
        self.buffer.append((s, a, r, s_next, float(done)))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = zip(*batch)
        to = lambda arr: torch.as_tensor(np.stack(arr), dtype=torch.float32, device=device)
        return (
            to(s),
            to(a),
            torch.as_tensor(r, dtype=torch.float32, device=device).unsqueeze(-1),
            to(s_next),
            torch.as_tensor(d, dtype=torch.float32, device=device).unsqueeze(-1),
        )
