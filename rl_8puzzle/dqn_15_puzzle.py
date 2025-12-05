import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_8puzzle.n_puzzle_env import NPuzzleEnv


class DQNTiles(nn.Module):
    def __init__(self, board_size: int = 4, embed_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.board_size = board_size
        self.num_tiles = board_size * board_size  # 16
        self.num_actions = 4  # up, down, left, right

        # Embed tile IDs (0..15) into vectors
        self.embedding = nn.Embedding(self.num_tiles, embed_dim)

        input_dim = self.num_tiles * embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 16) integer tile IDs
        returns: (batch, 4) Q-values
        """
        # shape: (batch, 16, embed_dim)
        emb = self.embedding(x.long())
        # flatten
        emb = emb.view(emb.size(0), -1)
        return self.net(emb)
