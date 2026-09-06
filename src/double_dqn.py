from __future__ import annotations
import argparse
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
from industrial_inventory_env import IndustrialInventoryEnv, generate_student_config
from .common import encode_observation, seed_everything
from .networks import FactorizedQNetwork

REWARD_SCALE = 0.1


class Replay:
    def __init__(self, capacity=200000):
        self.data = deque(maxlen=capacity)

    def add(self, *x):
        self.data.append(x)

    def sample(self, n, rng):
        idx = rng.choice(len(self.data), size=n, replace=False)
        batch = [self.data[i] for i in idx]
        return list(zip(*batch))

    def __len__(self):
        return len(self.data)


def chosen_q(q_all, actions):
    """q_all: [B,3,11], actions: [B,3] -> summed joint Q [B]."""
    per_product = q_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)
    return per_product.sum(dim=1)


def train(student_config, total_steps=700_000, batch_size=256, gamma=0.99,
          lr=2e-4, learning_starts=10000, target_update=2500,
          seed=51, hidden=256):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    online = FactorizedQNetwork(hidden)
    target = FactorizedQNetwork(hidden)
    target.load_state_dict(online.state_dict())
    target.eval()

    opt = torch.optim.Adam(online.parameters(), lr=lr)
    replay = Replay()
    env = IndustrialInventoryEnv(student_config, scenario_mode='random', domain_randomization=True)
    obs, _ = env.reset(seed=seed)
    episode_seed = seed

    for step in range(total_steps):
        eps_fraction = min(step / max(0.70 * total_steps, 1), 1.0)
        eps = 1.0 + eps_fraction * (0.05 - 1.0)
        s = encode_observation(obs)

        if rng.random() < eps:
            action = rng.integers(0, 11, size=3, dtype=np.int64)
        else:
            with torch.no_grad():
                q = online(torch.from_numpy(s).unsqueeze(0))[0]
                action = q.argmax(dim=1).numpy().astype(np.int64)

        nxt, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        replay.add(s, action.copy(), float(reward) * REWARD_SCALE,
                   encode_observation(nxt), float(done))
        obs = nxt

        if done:
            episode_seed += 1
            obs, _ = env.reset(seed=episode_seed)

        if len(replay) >= max(batch_size, learning_starts):
            S, A, R, N, D = replay.sample(batch_size, rng)
            S = torch.tensor(np.asarray(S), dtype=torch.float32)
            A = torch.tensor(np.asarray(A), dtype=torch.long)
            R = torch.tensor(R, dtype=torch.float32)
            N = torch.tensor(np.asarray(N), dtype=torch.float32)
            D = torch.tensor(D, dtype=torch.float32)

            pred = chosen_q(online(S), A)

            with torch.no_grad():
                # Double DQN: online selects each component, target evaluates it.
                next_actions = online(N).argmax(dim=2)
                next_q = chosen_q(target(N), next_actions)
                y = R + gamma * (1.0 - D) * next_q

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
    ap.add_argument('--roll-number', required=True)
    ap.add_argument('--steps', type=int, default=700000)
    ap.add_argument('--out', default='artifacts/double_dqn.pt')
    args = ap.parse_args()
    cfg = generate_student_config(args.roll_number)
    net = train(cfg, total_steps=args.steps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out)


if __name__ == '__main__':
    main()
