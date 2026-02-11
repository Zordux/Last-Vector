from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from last_vector_env import LastVectorEnv


class JsonlMetricsCallback(BaseCallback):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        row = {
            "ts": time.time(),
            "timesteps": int(self.num_timesteps),
            "fps": float(self.model.logger.name_to_value.get("time/fps", 0.0)),
            "ep_rew_mean": float(self.model.logger.name_to_value.get("rollout/ep_rew_mean", 0.0)),
            "ep_len_mean": float(self.model.logger.name_to_value.get("rollout/ep_len_mean", 0.0)),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path("runs") / args.run_id
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tensorboard"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "algo": "PPO",
        "policy": "MultiInputPolicy",
        "total_steps": args.total_steps,
        "seed": args.seed,
        "lr": args.lr,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_env = DummyVecEnv([lambda: Monitor(LastVectorEnv(render_mode="none"))])
    eval_env = DummyVecEnv([lambda: Monitor(LastVectorEnv(render_mode="none"))])

    model = PPO(
        "MultiInputPolicy",
        train_env,
        seed=args.seed,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        tensorboard_log=str(tb_dir),
        verbose=1,
    )

    callbacks = [
        CheckpointCallback(save_freq=50_000, save_path=str(ckpt_dir), name_prefix="ppo_last_vector"),
        EvalCallback(eval_env, best_model_save_path=str(run_dir), eval_freq=25_000, deterministic=True),
        JsonlMetricsCallback(run_dir / "metrics.jsonl"),
    ]

    model.learn(total_timesteps=args.total_steps, callback=callbacks, progress_bar=True)
    model.save(run_dir / "final_model")


if __name__ == "__main__":
    main()
