from __future__ import annotations
import argparse, random
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
from industrial_inventory_env import IndustrialInventoryEnv, generate_student_config
from .common import (
    encode_observation,
    joint_to_action_components,
    action_components_to_joint,
    valid_quantities,
    seed_everything,
)
from .networks import JointQNetwork


class Replay:
    def __init__(self, capacity=120000):
        self.data = deque(maxlen=capacity)

    def add(self, *x):
        self.data.append(x)

    def sample(self, n, rng):
        idx = rng.choice(len(self.data), size=n, replace=False)
        batch = [self.data[i] for i in idx]
        return list(zip(*batch))

    def __len__(self):
        return len(self.data)


def train(
    student_config,
    total_steps=350_000,
    batch_size=128,
    gamma=0.99,
    lr=2e-4,
    learning_starts=5000,
    target_update=2000,
    seed=51,
    hidden=256,
):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    online = JointQNetwork(hidden)
    target = JointQNetwork(hidden)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    replay = Replay()
    env = IndustrialInventoryEnv(
        student_config, scenario_mode="random", domain_randomization=True
    )
    obs, _ = env.reset(seed=seed)
    episode_seed = seed
    for step in range(total_steps):
        eps = max(0.05, 1.0 - 0.95 * step / (0.65 * total_steps))
        s = encode_observation(obs)
        if rng.random() < eps:
            proposed = rng.integers(0, 11, size=3, dtype=np.int64)
        else:
            with torch.no_grad():
                joint = int(
                    torch.argmax(online(torch.from_numpy(s).unsqueeze(0)), dim=1).item()
                )
                proposed = joint_to_action_components(joint)
        q = valid_quantities(obs, proposed)
        actual = env.quantities_to_action_indices(q)
        joint_actual = action_components_to_joint(actual)
        nxt, reward, term, trunc, _ = env.step(actual)
        done = term or trunc
        replay.add(s, joint_actual, float(reward), encode_observation(nxt), float(done))
        obs = nxt
        if done:
            episode_seed += 1
            obs, _ = env.reset(seed=episode_seed)
        if len(replay) >= max(batch_size, learning_starts):
            S, A, R, N, D = replay.sample(batch_size, rng)
            S = torch.tensor(np.asarray(S), dtype=torch.float32)
            A = torch.tensor(A, dtype=torch.long)
            R = torch.tensor(R, dtype=torch.float32)
            N = torch.tensor(np.asarray(N), dtype=torch.float32)
            D = torch.tensor(D, dtype=torch.float32)
            pred = online(S).gather(1, A[:, None]).squeeze(1)
            with torch.no_grad():
                next_a = online(N).argmax(dim=1)
                next_q = target(N).gather(1, next_a[:, None]).squeeze(1)
                y = R + gamma * (1 - D) * next_q
            loss = nn.functional.smooth_l1_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
            opt.step()
        if step > 0 and step % target_update == 0:
            target.load_state_dict(online.state_dict())
    return online


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll-number", required=True)
    ap.add_argument("--steps", type=int, default=350000)
    ap.add_argument("--out", default="artifacts/double_dqn.pt")
    args = ap.parse_args()
    cfg = generate_student_config(args.roll_number)
    net = train(cfg, total_steps=args.steps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out)


if __name__ == "__main__":
    main()
