from __future__ import annotations

import argparse
import statistics

from stable_baselines3 import PPO

from last_vector_env import LastVectorEnv
from last_vector_env.env import EnvConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--episode-seconds", type=float, default=180.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = LastVectorEnv(config=EnvConfig(episode_limit_s=args.episode_seconds), render_mode="none")
    model = PPO.load(args.model)

    ep_rewards = []
    ep_lengths = []
    ep_kills = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        final_info = {}

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            final_info = info

        ep_rewards.append(total_reward)
        ep_lengths.append(steps)
        ep_kills.append(float(final_info.get("kills", 0.0)))

        print(
            f"episode={ep} reward={total_reward:.3f} steps={steps} "
            f"kills={final_info.get('kills', 0)} damage={final_info.get('damage_taken', 0.0):.2f}"
        )

    print("\n=== aggregate ===")
    print(f"episodes: {args.episodes}")
    print(f"reward_mean: {statistics.fmean(ep_rewards):.4f}")
    print(f"steps_mean: {statistics.fmean(ep_lengths):.2f}")
    print(f"kills_mean: {statistics.fmean(ep_kills):.3f}")
    env.close()


if __name__ == "__main__":
    main()
