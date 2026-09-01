# RL Inventory Control — Five Distinct Techniques

This solution is built for the supplied `industrial_inventory_env` package and implements five distinct techniques:

1. Tabular Q-Learning
2. REINFORCE with learned baseline
3. A2C
4. PPO
5. Double DQN

## Important before training

Copy the supplied `industrial_inventory_env/` folder into this project root (or install it so Python can import it), then use **your own official roll number**. Do not train/report using the demo roll number.

## Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train all five

```bash
python train_all.py --roll-number YOUR_OFFICIAL_ROLL_NUMBER --output-dir artifacts
```

For a faster pipeline check:

```bash
python train_all.py --roll-number YOUR_OFFICIAL_ROLL_NUMBER --output-dir artifacts --smoke
```

The full defaults are intentionally moderate. For leaderboard work, increase training via each trainer's CLI options and compare on a held-out seed/scenario grid.

## Validate exported policies

After training, copy the generated model artifacts next to the corresponding policy file (the trainer does this automatically when `--submission-dir submissions` is used), then run:

```bash
python policy_validation_tests.py submissions/policy_tabular_q.py
python policy_validation_tests.py submissions/policy_reinforce.py
python policy_validation_tests.py submissions/policy_a2c.py
python policy_validation_tests.py submissions/policy_ppo.py
python policy_validation_tests.py submissions/policy_double_dqn.py
```

## Local validation

```bash
python evaluate_all.py --roll-number YOUR_OFFICIAL_ROLL_NUMBER --artifact-dir artifacts
```

The evaluator uses deterministic actions, official unscaled episode cost, multiple declared scenario families, and held-out seeds.

## Files

- `src/common.py`: state encoding, action mapping, seeding, capacity projection, evaluation helpers
- `src/networks.py`: PyTorch policy/value/Q networks
- `src/tabular_q.py`: Tabular Q-Learning
- `src/reinforce.py`: REINFORCE + baseline
- `src/a2c.py`: Advantage Actor-Critic
- `src/ppo.py`: PPO with GAE and clipped objective
- `src/double_dqn.py`: replay-buffer Double DQN
- `train_all.py`: end-to-end training/export driver
- `evaluate_all.py`: apples-to-apples local validation
- `submissions/policy_*.py`: required `run_policy(observation)` interfaces
- `inventory_5_models.ipynb`: report-friendly notebook orchestration

## Methodology notes

- Neural policies use the full documented observation after deterministic normalization.
- Tabular Q-Learning uses a deliberately coarser state discretization to make table reuse feasible.
- All policies use deterministic inference.
- A shared capacity projection only scales an already chosen order downward when current inventory + outstanding pipeline + requested order is conservatively above warehouse capacity. It never creates new non-zero product orders.
- Training uses the official environment reward by default; no hidden information is used.
