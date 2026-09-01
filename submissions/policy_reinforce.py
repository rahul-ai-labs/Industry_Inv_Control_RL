import os
import numpy as np
import torch
from torch import nn

PRODUCT_VOLUMES = np.asarray([2.0, 3.0, 1.5], dtype=np.float32)


def _mean_demand(o):
    h = np.asarray(o["demand_history"], dtype=np.float32)
    fb = np.asarray([30.0, 25.0, 35.0], dtype=np.float32)
    z = []
    for p in range(3):
        v = h[:, p]
        v = v[v > 0]
        z.append(v.mean() if v.size else fb[p])
    return np.asarray(z, dtype=np.float32)


def _enc(o):
    inv = np.asarray(o["inventory"], dtype=np.float32)
    pipe = np.asarray(o["arrival_pipeline"], dtype=np.float32)
    hist = np.asarray(o["demand_history"], dtype=np.float32)
    day = np.asarray(o["day"], dtype=np.float32)
    util = np.asarray(o["capacity_utilisation"], dtype=np.float32)
    recent = _mean_demand(o)
    pos = inv + pipe.sum(axis=1)
    return np.concatenate(
        [
            np.clip(inv / 250.0, 0, 4),
            np.clip(pipe.reshape(-1) / 100.0, 0, 5),
            np.clip(hist.reshape(-1) / 100.0, 0, 5),
            np.clip(day / 50.0, 0, 1),
            np.clip(util, 0, 1.5),
            np.clip(recent / 60.0, 0, 3),
            np.clip(pos / 300.0, 0, 4),
        ]
    ).astype(np.float32)


def _project(o, q):
    q = np.clip((np.asarray(q, dtype=np.int64) // 10) * 10, 0, 100)
    inv = np.asarray(o["inventory"], dtype=np.float32)
    pipe = np.asarray(o["arrival_pipeline"], dtype=np.float32).sum(axis=1)
    budget = max(0.0, 980.0 - float(np.dot(inv + pipe, PRODUCT_VOLUMES)))
    rv = float(np.dot(q, PRODUCT_VOLUMES))
    if rv <= budget + 1e-9 or rv <= 0:
        return q
    return np.clip(np.floor((q * (budget / rv)) / 10.0) * 10.0, 0, 100).astype(np.int64)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(44, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.actor = nn.Linear(128, 33)
        self.baseline = nn.Linear(128, 1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h).view(-1, 3, 11), self.baseline(h).squeeze(-1)


_net = Net()
_net.load_state_dict(
    torch.load(
        os.path.join(os.path.dirname(__file__), "reinforce.pt"),
        map_location="cpu",
        weights_only=True,
    )
)
_net.eval()


def run_policy(observation):
    with torch.no_grad():
        logits, _ = _net(torch.from_numpy(_enc(observation)).unsqueeze(0))
        a = logits.squeeze(0).argmax(dim=1).numpy()
    return _project(observation, a * 10).astype(int).tolist()
