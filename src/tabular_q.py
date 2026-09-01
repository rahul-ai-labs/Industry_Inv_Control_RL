from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from industrial_inventory_env import IndustrialInventoryEnv, generate_student_config
from .common import discretize_observation, valid_quantities, seed_everything


def train(
    student_config,
    episodes=6000,
    alpha=0.12,
    gamma=0.98,
    eps_start=1.0,
    eps_end=0.04,
    seed=11,
):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    q = defaultdict(lambda: np.zeros((3, 11), dtype=np.float32))
    env = IndustrialInventoryEnv(
        student_config, scenario_mode="random", domain_randomization=True
    )
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        eps = eps_end + (eps_start - eps_end) * np.exp(-5.0 * ep / max(episodes, 1))
        while True:
            s = discretize_observation(obs)
            if rng.random() < eps:
                a = rng.integers(0, 11, size=3, dtype=np.int64)
            else:
                a = np.argmax(q[s], axis=1).astype(np.int64)
            quantities = valid_quantities(obs, a)
            actual_a = env.quantities_to_action_indices(quantities)
            nxt, reward, terminated, truncated, _ = env.step(actual_a)
            s2 = discretize_observation(nxt)
            done = terminated or truncated
            # Cooperative independent heads: same environment reward updates each product's chosen action.
            for p in range(3):
                target = reward if done else reward + gamma * float(np.max(q[s2][p]))
                q[s][p, actual_a[p]] += alpha * (target - q[s][p, actual_a[p]])
            obs = nxt
            if done:
                break
    return dict(q)


def save_table(q, path):
    states = np.asarray(list(q.keys()), dtype=np.int16)
    values = (
        np.stack([q[tuple(s)] for s in states], axis=0).astype(np.float32)
        if len(states)
        else np.zeros((0, 3, 11), np.float32)
    )
    np.savez_compressed(path, states=states, q_values=values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll-number", required=True)
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--out", default="artifacts/tabular_q.npz")
    args = ap.parse_args()
    cfg = generate_student_config(args.roll_number)
    q = train(cfg, episodes=args.episodes)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_table(q, args.out)


if __name__ == "__main__":
    main()
