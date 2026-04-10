# LunarLander PPO Benchmark with autoexp

Autonomous ML experiment tracking demo: solving LunarLander-v3 with PPO using [autoexp](https://github.com/anthropics/claude-code/tree/main/.skills/autoexp) (Claude Code skill for autonomous ML research).

## Results

**Goal**: mean_reward >= 200, solved_rate >= 0.8 over 100 eval episodes.

| Experiment | Hypothesis | Status | mean_reward | solved_rate |
|-----------|-----------|--------|-------------|-------------|
| auto_001 | Baseline: default PPO, 100k steps | failed (stopped) | - | - |
| auto_002 | Wider net [256,256] + 300k steps | timeout | - | - |
| auto_003 | Smaller net [64,64] + lr=7e-4 | timeout | - | - |
| **auto_004** | **n_envs 8 + lr=1e-3 + 300k steps** | **completed** | **252.3** | **0.950** |

## Winning Config

```yaml
net_arch: [64, 64]
activation: tanh
learning_rate: 0.001
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.999
gae_lambda: 0.98
ent_coef: 0.01
n_envs: 8
total_timesteps: 300000
```

## Setup

```bash
uv sync
```

### Training

```bash
uv run python train.py
```

### Evaluation

```bash
uv run python eval.py runs/latest/model 100
```

### autoexp skill

Install the [autoexp](https://github.com/anthropics/claude-code/tree/main/.skills/autoexp) Claude Code skill to run autonomous experiments. Experiment history is tracked in `.autoexp/experiments.db`.
