from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable

import numpy as np
import torch

PRODUCT_VOLUMES = np.asarray([2.0, 3.0, 1.5], dtype=np.float32)
CAPACITY = 1000.0
N_PRODUCTS = 3
N_LEVELS = 11
N_JOINT_ACTIONS = N_LEVELS**N_PRODUCTS


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def action_components_to_joint(action: Iterable[int]) -> int:
    a = np.asarray(list(action), dtype=np.int64)
    if a.shape != (3,) or np.any(a < 0) or np.any(a > 10):
        raise ValueError("action must be three indices in [0, 10]")
    return int(a[0] * 121 + a[1] * 11 + a[2])


def joint_to_action_components(index: int) -> np.ndarray:
    if index < 0 or index >= N_JOINT_ACTIONS:
        raise ValueError("joint action outside [0, 1330]")
    a0 = index // 121
    rem = index % 121
    a1 = rem // 11
    a2 = rem % 11
    return np.asarray([a0, a1, a2], dtype=np.int64)


def indices_to_quantities(action: Iterable[int]) -> np.ndarray:
    a = np.asarray(list(action), dtype=np.int64)
    return a * 10


def quantities_to_indices(q: Iterable[int]) -> np.ndarray:
    q = np.asarray(list(q), dtype=np.int64)
    q = np.clip(np.rint(q / 10.0), 0, 10).astype(np.int64)
    return q


def capacity_project(
    observation: dict, quantities: Iterable[int], utilisation_limit: float = 0.98
) -> np.ndarray:
    """Conservatively scale down an RL-selected order if committed stock is too large.

    This post-processor never creates a new order. It only reduces requested quantities,
    preserving relative proportions as much as 10-unit granularity allows.
    """
    q = np.asarray(list(quantities), dtype=np.int64)
    q = np.clip((q // 10) * 10, 0, 100)
    inv = np.asarray(observation["inventory"], dtype=np.float32)
    pipe = np.asarray(observation["arrival_pipeline"], dtype=np.float32).sum(axis=1)
    committed_volume = float(np.dot(inv + pipe, PRODUCT_VOLUMES))
    budget = max(0.0, utilisation_limit * CAPACITY - committed_volume)
    requested_volume = float(np.dot(q, PRODUCT_VOLUMES))
    if requested_volume <= budget + 1e-9 or requested_volume <= 0:
        return q
    scale = budget / requested_volume
    q2 = (np.floor((q * scale) / 10.0) * 10.0).astype(np.int64)
    return np.clip(q2, 0, 100)


def valid_quantities(observation: dict, action_indices: Iterable[int]) -> np.ndarray:
    return capacity_project(observation, indices_to_quantities(action_indices))


def mean_recent_demand(observation: dict) -> np.ndarray:
    hist = np.asarray(observation["demand_history"], dtype=np.float32)
    out = np.zeros(3, dtype=np.float32)
    fallback = np.asarray([30.0, 25.0, 35.0], dtype=np.float32)
    for p in range(3):
        vals = hist[:, p]
        vals = vals[vals > 0]
        out[p] = vals.mean() if vals.size else fallback[p]
    return out


def encode_observation(observation: dict) -> np.ndarray:
    """44 deterministic normalized features from only documented observations."""
    inv = np.asarray(observation["inventory"], dtype=np.float32)
    pipe = np.asarray(observation["arrival_pipeline"], dtype=np.float32)
    hist = np.asarray(observation["demand_history"], dtype=np.float32)
    day = np.asarray(observation["day"], dtype=np.float32)
    util = np.asarray(observation["capacity_utilisation"], dtype=np.float32)
    recent = mean_recent_demand(observation)
    inv_pos = inv + pipe.sum(axis=1)
    x = np.concatenate(
        [
            np.clip(inv / 250.0, 0, 4),
            np.clip(pipe.reshape(-1) / 100.0, 0, 5),
            np.clip(hist.reshape(-1) / 100.0, 0, 5),
            np.clip(day / 50.0, 0, 1),
            np.clip(util, 0, 1.5),
            np.clip(recent / 60.0, 0, 3),
            np.clip(inv_pos / 300.0, 0, 4),
        ]
    ).astype(np.float32)
    assert x.shape == (44,), x.shape
    return x


def discretize_observation(observation: dict) -> tuple[int, ...]:
    """Compact state for tabular Q-Learning.

    Per product: inventory-position cover bucket and recent-demand bucket;
    plus episode phase and capacity bucket.
    """
    inv = np.asarray(observation["inventory"], dtype=np.float32)
    pipe = np.asarray(observation["arrival_pipeline"], dtype=np.float32).sum(axis=1)
    demand = mean_recent_demand(observation)
    cover = (inv + pipe) / np.maximum(demand, 1.0)
    cover_bucket = np.digitize(cover, bins=[0.75, 1.5, 2.25, 3.0, 4.0]).astype(int)
    demand_bucket = np.digitize(demand, bins=[22.5, 30.0, 37.5, 45.0]).astype(int)
    day = int(np.asarray(observation["day"]).reshape(-1)[0])
    phase = min(day // 10, 4)
    util = float(np.asarray(observation["capacity_utilisation"]).reshape(-1)[0])
    util_bucket = int(np.digitize([util], bins=[0.35, 0.55, 0.75, 0.9])[0])
    return tuple(cover_bucket.tolist() + demand_bucket.tolist() + [phase, util_bucket])


@dataclass
class EvalResult:
    scenario: str
    seed: int
    episode_cost: float


def evaluate_policy(
    student_config: dict,
    policy: Callable[[dict], list[int]],
    seeds: Iterable[int],
    scenarios: Iterable[str],
) -> list[EvalResult]:
    from industrial_inventory_env import IndustrialInventoryEnv

    results: list[EvalResult] = []
    for scenario in scenarios:
        for seed in seeds:
            env = IndustrialInventoryEnv(
                student_config, scenario_mode=scenario, domain_randomization=True
            )
            obs, _ = env.reset(seed=int(seed))
            while True:
                q = policy(obs)
                action = env.quantities_to_action_indices(q)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    results.append(
                        EvalResult(
                            str(scenario),
                            int(seed),
                            float(info["costs"]["episode_total"]),
                        )
                    )
                    break
    return results
