from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.distributions import Categorical
from industrial_inventory_env import IndustrialInventoryEnv, generate_student_config
from .common import encode_observation, seed_everything
from .networks import ActorCritic

REWARD_SCALE = 0.1


def train(student_config, total_steps=600_000, n_steps=32, gamma=0.99, lr=2e-4,
          value_coef=0.5, entropy_coef=0.005, seed=31, hidden=256):
    seed_everything(seed)
    net = ActorCritic(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    env = IndustrialInventoryEnv(student_config, scenario_mode='random', domain_randomization=True)

    obs, _ = env.reset(seed=seed)
    steps = 0
    episode_seed = seed

    while steps < total_steps:
        logps, values, rewards, dones, entropies = [], [], [], [], []

        for _ in range(n_steps):
            x = torch.from_numpy(encode_observation(obs)).unsqueeze(0)
            logits, value = net(x)
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()                       # EXACT action executed

            nxt, reward, term, trunc, _ = env.step(action.numpy())
            done = term or trunc

            logps.append(dist.log_prob(action).sum())    # log-prob of sampled action
            values.append(value.squeeze(0))
            rewards.append(float(reward) * REWARD_SCALE)
            dones.append(done)
            entropies.append(dist.entropy().sum())

            obs = nxt
            steps += 1
            if done:
                episode_seed += 1
                obs, _ = env.reset(seed=episode_seed)
            if steps >= total_steps:
                break

        with torch.no_grad():
            bootstrap = 0.0 if dones[-1] else float(
                net(torch.from_numpy(encode_observation(obs)).unsqueeze(0))[1].item()
            )

        returns = []
        g = bootstrap
        for r, d in zip(reversed(rewards), reversed(dones)):
            g = r + gamma * g * (0.0 if d else 1.0)
            returns.append(g)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)

        vals = torch.stack(values)
        adv = returns - vals
        norm_adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        actor_loss = -(torch.stack(logps) * norm_adv.detach()).mean()
        critic_loss = (adv ** 2).mean()
        entropy = torch.stack(entropies).mean()
        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()

    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roll-number', required=True)
    ap.add_argument('--steps', type=int, default=600000)
    ap.add_argument('--out', default='artifacts/a2c.pt')
    args = ap.parse_args()
    cfg = generate_student_config(args.roll_number)
    net = train(cfg, total_steps=args.steps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out)


if __name__ == '__main__':
    main()
