from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from last_vector_env import LastVectorEnv
from last_vector_env.env import EnvConfig


class CsvMetricsCallback(BaseCallback):
    """Append key scalar metrics to a CSV file during training."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wrote_header = self.path.exists() and self.path.stat().st_size > 0

    def _on_step(self) -> bool:
        row = {
            "ts": time.time(),
            "timesteps": int(self.num_timesteps),
            "fps": float(self.model.logger.name_to_value.get("time/fps", 0.0)),
            "ep_rew_mean": float(self.model.logger.name_to_value.get("rollout/ep_rew_mean", 0.0)),
            "ep_len_mean": float(self.model.logger.name_to_value.get("rollout/ep_len_mean", 0.0)),
        }

        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow(row)
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Last-Vector with PPO.")
    parser.add_argument("--run-id", default=None, help="Optional run identifier (default: timestamp).")
    parser.add_argument("--total-steps", type=int, default=2_000_000, help="Total training timesteps.")
    parser.add_argument("--episode-seconds", type=float, default=180.0, help="Episode time limit in seconds.")
    parser.add_argument("--seed", type=int, default=1337, help="Global deterministic seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Torch device.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout steps.")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO minibatch size.")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path("runs") / run_id
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tensorboard"
    metrics_path = run_dir / "metrics.csv"

    for directory in (run_dir, ckpt_dir, tb_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config = {
        "algo": "PPO",
        "policy": "MlpPolicy",
        "total_steps": int(args.total_steps),
        "seed": int(args.seed),
        "lr": float(args.lr),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "episode_seconds": float(args.episode_seconds),
        "device": args.device,
        "run_id": run_id,
    }
    write_json(run_dir / "config.json", config)

    def make_env(env_seed: int) -> Monitor:
        env_config = EnvConfig(episode_limit_s=args.episode_seconds, simulator_seed=env_seed)
        env = LastVectorEnv(config=env_config, render_mode="none")
        env.reset(seed=env_seed)
        return Monitor(env)

    train_env = DummyVecEnv([lambda: make_env(args.seed)])
    eval_env = DummyVecEnv([lambda: make_env(args.seed + 1)])

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        seed=args.seed,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        tensorboard_log=str(tb_dir),
        verbose=1,
        device=args.device,
    )

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=50_000,
                save_path=str(ckpt_dir),
                name_prefix="ppo_last_vector",
                save_replay_buffer=False,
                save_vecnormalize=False,
            ),
            EvalCallback(
                eval_env=eval_env,
                best_model_save_path=str(run_dir),
                log_path=str(run_dir),
                eval_freq=25_000,
                deterministic=True,
                render=False,
            ),
            CsvMetricsCallback(metrics_path),
        ]
    )

    model.learn(total_timesteps=args.total_steps, callback=callbacks, progress_bar=True)
    model.save(run_dir / "final_model")


if __name__ == "__main__":
    main()
