# LunarLander PPO Benchmark with autoexp

Autonomous ML experiment tracking demo: solving LunarLander-v3 with PPO using the [autoexp](https://github.com/MrTsepa/autoexp) Claude Code skill for autonomous ML research.

## How it works

The entire experiment loop was driven by giving Claude Code this prompt:

```
You are an autonomous ML research agent working on solving LunarLander-v3 with PPO.

Read .autoexp/program.md for your goal. The autoexp toolkit is at:
  python3 .claude/skills/autoexp/scripts/autoexp.py <command>

Your workflow each cycle:
1. python3 .claude/skills/autoexp/scripts/autoexp.py status
2. python3 .claude/skills/autoexp/scripts/autoexp.py results --last 5 --json
3. If training just finished: run eval, analyze, keep or revert
4. If nothing running: propose next experiment, edit config.yaml, validate, commit, train
5. python3 .claude/skills/autoexp/scripts/autoexp.py report > RESEARCH.md

Train command: uv run python train.py
Eval command: uv run python eval.py runs/latest/model 100

Keep experiments under 5 min. Start by checking the current state, then continue iterating.
```

This was invoked via the `/autoexp` skill command, which reads the research goal from `.autoexp/program.md`, then autonomously:

1. Reviews prior experiment results
2. Formulates a hypothesis for what to try next
3. Edits config/code, validates, and commits
4. Runs training and evaluation
5. Decides whether to keep or revert based on metrics

All 4 experiments — from the initial baseline to the solved configuration — were run in a single conversation with no manual intervention beyond the initial `/autoexp` call.

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

Install the [autoexp](https://github.com/MrTsepa/autoexp) Claude Code skill:

```bash
npx skills add MrTsepa/autoexp
```

Then initialize the project:

```bash
python3 .claude/skills/autoexp/scripts/autoexp.py init
```

Edit `.autoexp/program.md` with your research goal and `.autoexp/config.yaml` with editable/locked file rules. Experiment history is tracked in `.autoexp/experiments.db`.
