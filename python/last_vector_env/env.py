from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import last_vector_core


@dataclass
class EnvConfig:
    episode_limit_s: float = 180.0


class LastVectorEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["none"], "render_fps": 60}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: str = "none") -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.core = last_vector_core.Simulator(seed=0, episode_seconds=float(self.config.episode_limit_s))

        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, 0, 0, 0, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 2], dtype=np.float32),
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
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        s = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        obs = np.asarray(self.core.reset(s), dtype=np.float32)
        return np.ascontiguousarray(obs), {"seed": s}

    def step(self, action: np.ndarray):
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = self.core.step(action_arr)
        obs_np = np.asarray(obs, dtype=np.float32)
        return np.ascontiguousarray(obs_np), float(reward), bool(terminated), bool(truncated), dict(info)

    def render(self):
        return None

    def close(self):
        return None
