# Research Program

## Goal
Solve LunarLander-v3 with PPO. Target: mean_reward >= 200 over 100 evaluation episodes (solved_rate >= 0.8).

## Current Best
No experiments yet.

## Search Space
- Learning rate (1e-4 to 1e-2)
- Network architecture (net_arch: wider, deeper, different activations)
- PPO hyperparameters (n_steps, batch_size, n_epochs, clip_range)
- Discount factor (gamma: 0.99 to 0.9999)
- Entropy coefficient (ent_coef: 0.0 to 0.05)
- Training duration (total_timesteps: 50k to 500k)

## Constraints
- eval.py must not be modified
- Must train on CPU in reasonable time (<5 min per experiment)
- Must use gymnasium LunarLander-v3

## Known Insights
- LunarLander is typically solved with PPO in 200k-500k steps
- Default SB3 hyperparameters are a decent starting point
- Higher gamma (0.999) helps with long-horizon credit assignment
