from __future__ import annotations
import argparse, shutil
from pathlib import Path
import torch
from industrial_inventory_env import generate_student_config, public_config_summary
from src.tabular_q import train as train_q, save_table
from src.reinforce import train as train_reinforce
from src.a2c import train as train_a2c
from src.ppo import train as train_ppo
from src.double_dqn import train as train_ddqn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll-number", required=True)
    ap.add_argument("--output-dir", default="artifacts")
    ap.add_argument("--submission-dir", default="submissions")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = generate_student_config(args.roll_number)
    print("CONFIG:", public_config_summary(cfg))
    out = Path(args.output_dir)
    sub = Path(args.submission_dir)
    out.mkdir(parents=True, exist_ok=True)
    sub.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        qeps, reps, a2cs, ppos, ddqns = 8, 8, 300, 512, 500
    else:
        qeps, reps, a2cs, ppos, ddqns = 6000, 3500, 220000, 300000, 350000
    print("1/5 Tabular Q-Learning")
    q = train_q(cfg, episodes=qeps)
    save_table(q, out / "tabular_q.npz")
    print("2/5 REINFORCE + baseline")
    n = train_reinforce(cfg, episodes=reps)
    torch.save(n.state_dict(), out / "reinforce.pt")
    print("3/5 A2C")
    n = train_a2c(cfg, total_steps=a2cs)
    torch.save(n.state_dict(), out / "a2c.pt")
    print("4/5 PPO")
    n = train_ppo(cfg, total_steps=ppos, rollout_steps=min(1024, ppos))
    torch.save(n.state_dict(), out / "ppo.pt")
    print("5/5 Double DQN")
    n = train_ddqn(
        cfg, total_steps=ddqns, learning_starts=min(5000, max(64, ddqns // 4))
    )
    torch.save(n.state_dict(), out / "double_dqn.pt")
    for name in ["tabular_q.npz", "reinforce.pt", "a2c.pt", "ppo.pt", "double_dqn.pt"]:
        shutil.copy2(out / name, sub / name)
    print("Done. Artifacts copied next to policy files in", sub)


if __name__ == "__main__":
    main()
