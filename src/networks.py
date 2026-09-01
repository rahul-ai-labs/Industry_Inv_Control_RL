from __future__ import annotations
import torch
from torch import nn

INPUT_DIM = 44


class ActorCritic(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, 33)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h).view(-1, 3, 11), self.critic(h).squeeze(-1)


class PolicyBaseline(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, 33)
        self.baseline = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h).view(-1, 3, 11), self.baseline(h).squeeze(-1)


class JointQNetwork(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1331),
        )

    def forward(self, x):
        return self.net(x)
