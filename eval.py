"""Evaluate a trained LunarLander model. Deterministic, fixed seed, prints metrics."""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def evaluate(model_path: str, n_episodes: int = 100, seed: int = 0):
    model = PPO.load(model_path)
    rewards = []

    for i in range(n_episodes):
        env = gym.make("LunarLander-v3")
        obs, _ = env.reset(seed=seed + i)
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        env.close()

    mean = np.mean(rewards)
    std = np.std(rewards)
    solved = np.mean([r >= 200 for r in rewards])

    print(f"mean_reward: {mean:.1f}")
    print(f"std_reward: {std:.1f}")
    print(f"min_reward: {np.min(rewards):.1f}")
    print(f"max_reward: {np.max(rewards):.1f}")
    print(f"solved_rate: {solved:.3f}")
    print(f"episodes: {n_episodes}")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/latest/model"
    n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    evaluate(model_path, n_episodes)
