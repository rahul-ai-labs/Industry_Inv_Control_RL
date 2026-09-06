from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from industrial_inventory_env import IndustrialInventoryEnv, generate_student_config
from .common import discretize_observation, base_stock_fallback, seed_everything

REWARD_SCALE = 0.1


def train(student_config, episodes=12000, alpha=0.08, gamma=0.98,
          eps_start=1.0, eps_end=0.03, seed=11):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    q = defaultdict(lambda: np.zeros((3, 11), dtype=np.float32))
    visits = defaultdict(int)
    env = IndustrialInventoryEnv(student_config, scenario_mode='random', domain_randomization=True)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        eps = eps_end + (eps_start - eps_end) * np.exp(-5.0 * ep / max(episodes, 1))

        while True:
            s = discretize_observation(obs)
            visits[s] += 1

            if rng.random() < eps:
                action = rng.integers(0, 11, size=3, dtype=np.int64)
            else:
                # Use sensible prior behavior until this state has some experience.
                action = base_stock_fallback(obs) if visits[s] < 3 else np.argmax(q[s], axis=1).astype(np.int64)

            nxt, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            r = float(reward) * REWARD_SCALE
            s2 = discretize_observation(nxt)

            for p in range(3):
                target = r if done else r + gamma * float(np.max(q[s2][p]))
                q[s][p, action[p]] += alpha * (target - q[s][p, action[p]])

            obs = nxt
            if done:
                break

    return dict(q)


def save_table(q, path):
    states = np.asarray(list(q.keys()), dtype=np.int16)
    values = np.stack([q[tuple(s)] for s in states], axis=0).astype(np.float32) if len(states) else np.zeros((0, 3, 11), np.float32)
    np.savez_compressed(path, states=states, q_values=values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roll-number', required=True)
    ap.add_argument('--episodes', type=int, default=12000)
    ap.add_argument('--out', default='artifacts/tabular_q.npz')
    args = ap.parse_args()
    cfg = generate_student_config(args.roll_number)
    q = train(cfg, episodes=args.episodes)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_table(q, args.out)


if __name__ == '__main__':
    main()
