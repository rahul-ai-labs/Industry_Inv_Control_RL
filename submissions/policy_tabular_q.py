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


_MODEL = os.path.join(os.path.dirname(__file__), "tabular_q.npz")
_d = np.load(_MODEL)
_states = _d["states"]
_q = _d["q_values"]
_table = {tuple(map(int, s)): v for s, v in zip(_states, _q)}


def _state(o):
    inv = np.asarray(o["inventory"], dtype=np.float32)
    pipe = np.asarray(o["arrival_pipeline"], dtype=np.float32).sum(axis=1)
    demand = _mean_demand(o)
    cover = (inv + pipe) / np.maximum(demand, 1.0)
    cb = np.digitize(cover, [0.75, 1.5, 2.25, 3.0, 4.0]).astype(int)
    db = np.digitize(demand, [22.5, 30.0, 37.5, 45.0]).astype(int)
    day = int(np.asarray(o["day"]).reshape(-1)[0])
    phase = min(day // 10, 4)
    util = float(np.asarray(o["capacity_utilisation"]).reshape(-1)[0])
    ub = int(np.digitize([util], [0.35, 0.55, 0.75, 0.9])[0])
    return tuple(cb.tolist() + db.tolist() + [phase, ub])


def run_policy(observation):
    v = _table.get(_state(observation))
    a = np.zeros(3, dtype=np.int64) if v is None else np.argmax(v, axis=1)
    return _project(observation, a * 10).astype(int).tolist()
