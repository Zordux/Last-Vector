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
        self.episodes_done = 0
        self.last_episode_reward = 0.0
        self.last_episode_len = 0.0
        self.last_episode_kills = 0.0
        self.last_shots_fired = 0
        self.last_hits = 0
        self.last_accuracy = 0.0
        self.last_damage_dealt = 0.0
        self.last_damage_taken = 0.0
        self._file_handle = None
        self._csv_writer = None

    def _on_training_start(self) -> None:
        """Open CSV file and write header if needed."""
        try:
            self._file_handle = self.path.open("a", encoding="utf-8", newline="")
            fieldnames = [
                "ts",
                "timesteps",
                "fps",
                "ep_rew_mean",
                "ep_len_mean",
                "episodes_done",
                "last_ep_reward",
                "last_ep_len",
                "last_ep_kills",
                "kills",
                "shots_fired",
                "hits",
                "accuracy",
                "damage_dealt",
                "damage_taken",
            ]
            self._csv_writer = csv.DictWriter(self._file_handle, fieldnames=fieldnames)
            if not self._wrote_header:
                self._csv_writer.writeheader()
                self._file_handle.flush()
                self._wrote_header = True
        except Exception as e:
            if self._file_handle is not None:
                self._file_handle.close()
                self._file_handle = None
            self._csv_writer = None
            raise

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if episode is not None:
                self.episodes_done += 1
                self.last_episode_reward = float(episode.get("r", 0.0))
                self.last_episode_len = float(episode.get("l", 0.0))
                self.last_episode_kills = float(episode.get("kills", info.get("kills", 0.0)))
            self.last_shots_fired = int(info.get("shots_fired", self.last_shots_fired))
            self.last_hits = int(info.get("hits", self.last_hits))
            self.last_accuracy = float(info.get("accuracy", self.last_accuracy))
            self.last_damage_dealt = float(info.get("damage_dealt", self.last_damage_dealt))
            self.last_damage_taken = float(info.get("damage_taken", self.last_damage_taken))

        if self._csv_writer is None or self._file_handle is None:
            return True

        row = {
            "ts": time.time(),
            "timesteps": int(self.num_timesteps),
            "fps": float(self.model.logger.name_to_value.get("time/fps", 0.0)),
            "ep_rew_mean": float(self.model.logger.name_to_value.get("rollout/ep_rew_mean", 0.0)),
            "ep_len_mean": float(self.model.logger.name_to_value.get("rollout/ep_len_mean", 0.0)),
            "episodes_done": int(self.episodes_done),
            "last_ep_reward": float(self.last_episode_reward),
            "last_ep_len": float(self.last_episode_len),
            "last_ep_kills": float(self.last_episode_kills),
            "kills": float(self.last_episode_kills),
            "shots_fired": int(self.last_shots_fired),
            "hits": int(self.last_hits),
            "accuracy": float(self.last_accuracy),
            "damage_dealt": float(self.last_damage_dealt),
            "damage_taken": float(self.last_damage_taken),
        }

        self._csv_writer.writerow(row)
        self._file_handle.flush()
        return True

    def _on_training_end(self) -> None:
        """Close CSV file handle."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            self._csv_writer = None


class StatusCallback(BaseCallback):
    """Persist lightweight training status for dashboard polling."""

    def __init__(
        self,
        *,
        path: Path,
        total_steps: int,
        device: str,
        eval_callback: EvalCallback,
        metrics_callback: CsvMetricsCallback,
        update_every_seconds: float = 7.0,
        update_every_steps: int = 10_000,
    ):
        super().__init__()
        self.path = path
        self.total_steps = int(total_steps)
        self.device = device
        self.eval_callback = eval_callback
        self.metrics_callback = metrics_callback
        self.update_every_seconds = float(update_every_seconds)
        self.update_every_steps = int(update_every_steps)
        self._last_write_ts = 0.0
        self._last_write_steps = 0

    def _write_status(self, state: str) -> None:
        payload = {
            "state": state,
            "steps_done": int(self.num_timesteps),
            "total_steps": int(self.total_steps),
            "progress_pct": (100.0 * float(self.num_timesteps) / float(self.total_steps)) if self.total_steps > 0 else 0.0,
            "fps": float(self.model.logger.name_to_value.get("time/fps", 0.0)),
            "episodes_done": int(self.metrics_callback.episodes_done),
            "latest_reward": float(self.metrics_callback.last_episode_reward),
            "latest_ep_len": float(self.metrics_callback.last_episode_len),
            "latest_kills": float(self.metrics_callback.last_episode_kills),
            "best_model_path": str((self.path.parent / "best_model.zip").resolve()),
            "best_score": float(self.eval_callback.best_mean_reward),
            "device": str(self.model.device),
            "device_requested": self.device,
            "last_update_time": time.time(),
        }
        write_json(self.path, payload)

    def _on_step(self) -> bool:
        now = time.time()
        if (now - self._last_write_ts) >= self.update_every_seconds or (
            int(self.num_timesteps) - self._last_write_steps
        ) >= self.update_every_steps:
            self._write_status("running")
            self._last_write_ts = now
            self._last_write_steps = int(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        self._write_status("completed")


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


def validate_args(args: argparse.Namespace) -> None:
    if args.total_steps <= 0:
        raise ValueError("--total-steps must be > 0")
    if args.episode_seconds <= 0.0:
        raise ValueError("--episode-seconds must be > 0")
    if args.n_steps <= 0:
        raise ValueError("--n-steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_global_seed(args.seed)

    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path("runs") / run_id
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tensorboard"
    metrics_path = run_dir / "metrics.csv"

    for directory in (run_dir, ckpt_dir, tb_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_log = run_dir / "train.log"
    error_log = run_dir / "error.log"
    train_log.touch(exist_ok=True)
    error_log.touch(exist_ok=True)

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
    write_json(
        run_dir / "status.json",
        {
            "state": "running",
            "steps_done": 0,
            "total_steps": int(args.total_steps),
            "progress_pct": 0.0,
            "fps": 0.0,
            "episodes_done": 0,
            "latest_reward": 0.0,
            "latest_ep_len": 0.0,
            "latest_kills": 0.0,
            "best_model_path": str((run_dir / "best_model.zip").resolve()),
            "best_score": 0.0,
            "device": args.device,
            "last_update_time": time.time(),
        },
    )

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

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(ckpt_dir),
        name_prefix="ppo_last_vector",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=25_000,
        deterministic=True,
        render=False,
    )
    metrics_cb = CsvMetricsCallback(metrics_path)
    status_cb = StatusCallback(
        path=run_dir / "status.json",
        total_steps=args.total_steps,
        device=args.device,
        eval_callback=eval_cb,
        metrics_callback=metrics_cb,
    )
    callbacks = CallbackList([checkpoint_cb, eval_cb, metrics_cb, status_cb])

    try:
        model.learn(total_timesteps=args.total_steps, callback=callbacks, progress_bar=True)
        model.save(run_dir / "final_model")
    except Exception as exc:
        error_log.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        write_json(
            run_dir / "status.json",
            {
                "state": "failed",
                "steps_done": int(getattr(model, "num_timesteps", 0)),
                "total_steps": int(args.total_steps),
                "progress_pct": (100.0 * float(getattr(model, "num_timesteps", 0)) / float(args.total_steps)),
                "fps": float(model.logger.name_to_value.get("time/fps", 0.0)),
                "episodes_done": int(metrics_cb.episodes_done),
                "latest_reward": float(metrics_cb.last_episode_reward),
                "latest_ep_len": float(metrics_cb.last_episode_len),
                "latest_kills": float(metrics_cb.last_episode_kills),
                "best_model_path": str((run_dir / "best_model.zip").resolve()),
                "best_score": float(eval_cb.best_mean_reward),
                "device": str(model.device),
                "last_update_time": time.time(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


if __name__ == "__main__":
    main()
