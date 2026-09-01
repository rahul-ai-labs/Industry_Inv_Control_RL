from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.distributions import Categorical
from industrial_inventory_env import IndustrialInventoryEnv, generate_student_config
from .common import encode_observation, seed_everything
from .networks import PolicyBaseline

REWARD_SCALE = 0.1


def discounted_returns(rewards, gamma):
    out = []
    g = 0.0
    for r in reversed(rewards):
        g = float(r) + gamma * g
        out.append(g)
    return list(reversed(out))


def train(student_config, episodes=8000, gamma=0.99, lr=2e-4,
          entropy_coef=0.005, seed=21, hidden=256):
    seed_everything(seed)
    net = PolicyBaseline(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    env = IndustrialInventoryEnv(student_config, scenario_mode='random', domain_randomization=True)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        logps, vals, ents, rewards = [], [], [], []

        while True:
            x = torch.from_numpy(encode_observation(obs)).unsqueeze(0)
            logits, value = net(x)
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()

            logps.append(dist.log_prob(action).sum())
            ents.append(dist.entropy().sum())
            vals.append(value.squeeze(0))

            obs, reward, term, trunc, _ = env.step(action.numpy())
            rewards.append(float(reward) * REWARD_SCALE)
            if term or trunc:
                break

        ret = torch.tensor(discounted_returns(rewards, gamma), dtype=torch.float32)
        values = torch.stack(vals)
        adv = ret - values.detach()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        policy_loss = -(torch.stack(logps) * adv).mean()
        value_loss = 0.5 * ((values - ret) ** 2).mean()
        entropy_bonus = torch.stack(ents).mean()
        loss = policy_loss + value_loss - entropy_coef * entropy_bonus

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()

    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roll-number', required=True)
    ap.add_argument('--episodes', type=int, default=8000)
    ap.add_argument('--out', default='artifacts/reinforce.pt')
    args = ap.parse_args()
    cfg = generate_student_config(args.roll_number)
    net = train(cfg, episodes=args.episodes)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out)


if __name__ == '__main__':
    main()
