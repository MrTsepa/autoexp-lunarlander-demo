"""Train PPO on LunarLander-v3. Config-driven, outputs metrics to stdout."""

import sys
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback


class MetricLogger(BaseCallback):
    """Print metrics to stdout for autoexp to parse."""

    def __init__(self, log_interval=5000, verbose=0):
        super().__init__(verbose)
        self._log_interval = log_interval
        self._episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])

        if self.num_timesteps % self._log_interval < self.locals.get("n_steps", 2048):
            if self._episode_rewards:
                mean_r = sum(self._episode_rewards) / len(self._episode_rewards)
                print(f"step: {self.num_timesteps} mean_reward: {mean_r:.1f} episodes: {len(self._episode_rewards)}")
                sys.stdout.flush()
                self._episode_rewards = []
        return True


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    env = make_vec_env(cfg["env"], n_envs=cfg.get("n_envs", 4), seed=cfg.get("seed", 42))

    # Map activation name to torch module
    import torch.nn as nn
    activations = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}
    act_fn = activations.get(cfg.get("activation", "tanh"), nn.Tanh)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.get("learning_rate", 3e-4),
        n_steps=cfg.get("n_steps", 2048),
        batch_size=cfg.get("batch_size", 64),
        n_epochs=cfg.get("n_epochs", 10),
        gamma=cfg.get("gamma", 0.999),
        gae_lambda=cfg.get("gae_lambda", 0.98),
        clip_range=cfg.get("clip_range", 0.2),
        ent_coef=cfg.get("ent_coef", 0.01),
        vf_coef=cfg.get("vf_coef", 0.5),
        max_grad_norm=cfg.get("max_grad_norm", 0.5),
        policy_kwargs={"net_arch": cfg.get("net_arch", [64, 64]), "activation_fn": act_fn},
        seed=cfg.get("seed", 42),
        verbose=0,
    )

    total = cfg.get("total_timesteps", 100_000)
    print(f"Training {cfg['env']} with PPO for {total} steps")
    print(f"  net_arch: {cfg.get('net_arch')} activation: {cfg.get('activation')}")
    print(f"  lr: {cfg.get('learning_rate')} ent_coef: {cfg.get('ent_coef')} gamma: {cfg.get('gamma')}")
    sys.stdout.flush()

    model.learn(total_timesteps=total, callback=MetricLogger())

    run_dir = Path("runs") / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir / "model")
    print(f"model_saved: {run_dir / 'model'}")


if __name__ == "__main__":
    main()
