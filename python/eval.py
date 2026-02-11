from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from last_vector_env import LastVectorEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=2024)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = LastVectorEnv(render_mode="human")
    model = PPO.load(args.model)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
    env.close()


if __name__ == "__main__":
    main()
