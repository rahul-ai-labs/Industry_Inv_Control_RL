from __future__ import annotations
import torch
from torch import nn

INPUT_DIM = 56


class ActorCritic(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, 33)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h).view(-1, 3, 11), self.critic(h).squeeze(-1)


class PolicyBaseline(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, 33)
        self.baseline = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h).view(-1, 3, 11), self.baseline(h).squeeze(-1)


class FactorizedQNetwork(nn.Module):
    """Three 11-action Q heads; chosen-action values are summed for the joint decision."""
    def __init__(self, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 33)

    def forward(self, x):
        h = self.body(x)
        return self.head(h).view(-1, 3, 11)
