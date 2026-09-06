from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable

import numpy as np
import torch

N_PRODUCTS = 3
N_LEVELS = 11
REFERENCE_DEMAND = np.asarray([30.0, 25.0, 35.0], dtype=np.float32)
LEAD_TIMES = np.asarray([3.0, 2.0, 1.0], dtype=np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def indices_to_quantities(action: Iterable[int]) -> np.ndarray:
    a = np.asarray(list(action), dtype=np.int64)
    if a.shape != (3,) or np.any(a < 0) or np.any(a > 10):
        raise ValueError("action must be three indices in [0, 10]")
    return a * 10


def quantities_to_indices(q: Iterable[int]) -> np.ndarray:
    q = np.asarray(list(q), dtype=np.int64)
    return np.clip(np.rint(q / 10.0), 0, 10).astype(np.int64)


def demand_estimate(observation: dict) -> np.ndarray:
    """Recency-weighted demand estimate with a reference-demand prior early in an episode."""
    hist = np.asarray(observation["demand_history"], dtype=np.float32)
    day = int(np.asarray(observation["day"]).reshape(-1)[0])

    # Older -> newer weights. Zero-padded rows are excluded and weights renormalized.
    base_w = np.asarray([0.05, 0.07, 0.10, 0.13, 0.17, 0.21, 0.27], dtype=np.float32)
    estimates = np.empty(3, dtype=np.float32)

    for p in range(3):
        vals = hist[:, p]
        mask = vals > 0
        if not np.any(mask):
            estimates[p] = REFERENCE_DEMAND[p]
            continue
        w = base_w[mask]
        w = w / w.sum()
        recent = float(np.dot(vals[mask], w))
        confidence = min(int(mask.sum()), 7) / 7.0
        estimates[p] = confidence * recent + (1.0 - confidence) * REFERENCE_DEMAND[p]

    return np.maximum(estimates, 1.0)


def encode_observation(observation: dict) -> np.ndarray:
    """56 normalized features using only documented observation fields."""
    inv = np.asarray(observation["inventory"], dtype=np.float32)
    pipe = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    hist = np.asarray(observation["demand_history"], dtype=np.float32)
    day = np.asarray(observation["day"], dtype=np.float32).reshape(-1)
    util = np.asarray(observation["capacity_utilisation"], dtype=np.float32).reshape(-1)

    demand = demand_estimate(observation)
    inv_pos = inv + pipe.sum(axis=1)
    days_supply = inv_pos / np.maximum(demand, 1.0)
    lead_demand = demand * LEAD_TIMES
    gap = lead_demand - inv_pos
    next_arrivals = pipe[:, 0]

    x = np.concatenate([
        np.clip(inv / 200.0, 0, 3),                    # 3
        np.clip(pipe.reshape(-1) / 100.0, 0, 5),       # 12
        np.clip(hist.reshape(-1) / 80.0, 0, 5),        # 21
        np.clip(day / 50.0, 0, 1),                     # 1
        np.clip(util, 0, 1.5),                         # 1
        np.clip(demand / 60.0, 0, 3),                  # 3
        np.clip(inv_pos / 250.0, 0, 4),                # 3
        np.clip(days_supply / 5.0, 0, 3),              # 3
        np.clip(lead_demand / 150.0, 0, 3),            # 3
        np.clip(gap / 150.0, -3, 3),                   # 3
        np.clip(next_arrivals / 100.0, 0, 5),          # 3
    ]).astype(np.float32)

    assert x.shape == (56,), x.shape
    return x


def discretize_observation(observation: dict) -> tuple[int, ...]:
    """Compact tabular state emphasizing lead-time inventory adequacy."""
    inv = np.asarray(observation["inventory"], dtype=np.float32)
    pipe = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    demand = demand_estimate(observation)
    inv_pos = inv + pipe.sum(axis=1)

    # How many expected lead-time-demand multiples are covered?
    cover = inv_pos / np.maximum(demand * LEAD_TIMES, 1.0)
    cover_bucket = np.digitize(cover, bins=[0.5, 0.8, 1.0, 1.25, 1.6]).astype(int)
    demand_bucket = np.digitize(demand, bins=[22.5, 30.0, 37.5, 45.0]).astype(int)

    day = int(np.asarray(observation["day"]).reshape(-1)[0])
    phase = min(max((day - 1) // 10, 0), 4)
    util = float(np.asarray(observation["capacity_utilisation"]).reshape(-1)[0])
    util_bucket = int(np.digitize([util], bins=[0.35, 0.55, 0.75, 0.9])[0])

    return tuple(cover_bucket.tolist() + demand_bucket.tolist() + [phase, util_bucket])


def base_stock_fallback(observation: dict) -> np.ndarray:
    """Deterministic fallback for unseen tabular states; returns action indices 0..10."""
    inv = np.asarray(observation["inventory"], dtype=np.float32)
    pipe = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    demand = demand_estimate(observation)
    inv_pos = inv + pipe.sum(axis=1)

    # Lead-time demand plus a modest safety buffer for uncertainty/delay.
    target = demand * (LEAD_TIMES + 0.75)
    needed = np.maximum(target - inv_pos, 0.0)
    quantities = np.clip(np.ceil(needed / 10.0) * 10.0, 0, 100)
    return (quantities / 10.0).astype(np.int64)


@dataclass
class EvalResult:
    scenario: str
    seed: int
    episode_cost: float


def evaluate_policy(student_config: dict, policy: Callable[[dict], list[int]], seeds: Iterable[int], scenarios: Iterable[str]) -> list[EvalResult]:
    from industrial_inventory_env import IndustrialInventoryEnv

    results: list[EvalResult] = []
    for scenario in scenarios:
        for seed in seeds:
            env = IndustrialInventoryEnv(student_config, scenario_mode=scenario, domain_randomization=True)
            obs, _ = env.reset(seed=int(seed))
            while True:
                quantities = policy(obs)
                action = env.quantities_to_action_indices(quantities)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    results.append(EvalResult(str(scenario), int(seed), float(info["costs"]["episode_total"])))
                    break
    return results
