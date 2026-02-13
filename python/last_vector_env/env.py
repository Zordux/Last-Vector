from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import last_vector_core


@dataclass
class EnvConfig:
    """Runtime configuration for LastVectorEnv."""

    episode_limit_s: float = 180.0
    simulator_seed: int = 0


class LastVectorEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium wrapper around the native Last-Vector simulator."""

    metadata = {"render_modes": ["none"], "render_fps": 60}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: str = "none") -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.core = last_vector_core.Simulator(
            seed=int(self.config.simulator_seed),
            episode_seconds=float(self.config.episode_limit_s),
        )

        self.action_space = spaces.Box(
            low=last_vector_core.Simulator.action_low(),
            high=last_vector_core.Simulator.action_high(),
            dtype=np.float32,
        )

        obs_dim = int(self.core.obs_dim())
        self.observation_space = spaces.Box(
            low=np.full((obs_dim,), -np.inf, dtype=np.float32),
            high=np.full((obs_dim,), np.inf, dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment state and return (observation, info)."""

        del options
        super().reset(seed=seed)
        resolved_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        obs = np.asarray(self.core.reset(int(resolved_seed)), dtype=np.float32)
        return np.ascontiguousarray(obs), {"seed": int(resolved_seed)}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance one step and return Gymnasium step tuple."""

        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        action_arr = np.clip(action_arr, self.action_space.low, self.action_space.high)
        obs, reward, terminated, truncated, info = self.core.step(action_arr)
        obs_np = np.asarray(obs, dtype=np.float32)
        return (
            np.ascontiguousarray(obs_np),
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info),
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
