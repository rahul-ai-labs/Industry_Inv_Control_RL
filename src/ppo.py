from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from industrial_inventory_env import IndustrialInventoryEnv, generate_student_config
from .common import encode_observation, valid_quantities, seed_everything
from .networks import ActorCritic


def train(
    student_config,
    total_steps=300_000,
    rollout_steps=1024,
    epochs=8,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    lr=3e-4,
    value_coef=0.5,
    entropy_coef=0.002,
    seed=41,
    hidden=128,
):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    net = ActorCritic(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    env = IndustrialInventoryEnv(
        student_config, scenario_mode="random", domain_randomization=True
    )
    obs, _ = env.reset(seed=seed)
    episode_seed = seed
    steps = 0
    while steps < total_steps:
        xs = []
        acts = []
        old_logps = []
        rewards = []
        dones = []
        values = []
        for _ in range(min(rollout_steps, total_steps - steps)):
            x_np = encode_observation(obs)
            x = torch.from_numpy(x_np).unsqueeze(0)
            with torch.no_grad():
                logits, value = net(x)
                dist = Categorical(logits=logits.squeeze(0))
                sampled = dist.sample()
                quantities = valid_quantities(obs, sampled.numpy())
                actual = env.quantities_to_action_indices(quantities)
                at = torch.as_tensor(actual, dtype=torch.long)
                lp = dist.log_prob(at).sum()
            nxt, reward, term, trunc, _ = env.step(actual)
            done = term or trunc
            xs.append(x_np)
            acts.append(actual.copy())
            old_logps.append(float(lp))
            rewards.append(float(reward))
            dones.append(done)
            values.append(float(value.item()))
            obs = nxt
            steps += 1
            if done:
                episode_seed += 1
                obs, _ = env.reset(seed=episode_seed)
        with torch.no_grad():
            next_value = (
                0.0
                if dones[-1]
                else float(
                    net(torch.from_numpy(encode_observation(obs)).unsqueeze(0))[
                        1
                    ].item()
                )
            )
        adv = np.zeros(len(rewards), dtype=np.float32)
        last = 0.0
        for t in reversed(range(len(rewards))):
            nv = next_value if t == len(rewards) - 1 else values[t + 1]
            nonterminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + gamma * nv * nonterminal - values[t]
            last = delta + gamma * gae_lambda * nonterminal * last
            adv[t] = last
        ret = adv + np.asarray(values, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        X = torch.tensor(np.asarray(xs), dtype=torch.float32)
        A = torch.tensor(np.asarray(acts), dtype=torch.long)
        OLD = torch.tensor(old_logps, dtype=torch.float32)
        ADV = torch.tensor(adv)
        RET = torch.tensor(ret)
        n = len(xs)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                ix = torch.tensor(order[start : start + batch_size], dtype=torch.long)
                logits, v = net(X[ix])
                dist = Categorical(logits=logits)
                lp = dist.log_prob(A[ix]).sum(dim=1)
                ratio = torch.exp(lp - OLD[ix])
                s1 = ratio * ADV[ix]
                s2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * ADV[ix]
                loss = (
                    -torch.minimum(s1, s2).mean()
                    + value_coef * ((v - RET[ix]) ** 2).mean()
                    - entropy_coef * dist.entropy().sum(dim=1).mean()
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.8)
                opt.step()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll-number", required=True)
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--out", default="artifacts/ppo.pt")
    args = ap.parse_args()
    cfg = generate_student_config(args.roll_number)
    net = train(cfg, total_steps=args.steps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out)


if __name__ == "__main__":
    main()
