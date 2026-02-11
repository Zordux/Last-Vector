from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .bridge import SimulatorBridge
from .reward import shaped_reward


@dataclass
class EnvConfig:
    episode_limit_s: float = 180.0


class LastVectorEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["none"], "render_fps": 60}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: str = "none") -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.bridge = SimulatorBridge()

        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, 0, 0, 0, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 2], dtype=np.float32),
            shape=(self.bridge.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.bridge.obs_dim,),
            dtype=np.float32,
        )

        self._last_info: Dict[str, Any] = {}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        use_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        obs = self.bridge.reset(use_seed)
        self._last_info = {
            "kills": 0.0,
            "damage_taken": 0.0,
            "shots_fired": 0.0,
            "shots_hit": 0.0,
            "nearest_zombie_distance": 9999.0,
        }
        return obs, {"seed": use_seed}

    def step(self, action: np.ndarray):
        out = self.bridge.step(action)

        info = out.info
        kills_delta = float(info.get("kills", 0.0) - self._last_info.get("kills", 0.0))
        damage_delta = float(info.get("damage_taken", 0.0) - self._last_info.get("damage_taken", 0.0))
        shots_delta = float(info.get("shots_fired", 0.0) - self._last_info.get("shots_fired", 0.0))
        hits_delta = float(info.get("shots_hit", 0.0) - self._last_info.get("shots_hit", 0.0))
        nearest = float(info.get("nearest_zombie_distance", 9999.0))

        reward = shaped_reward(
            base_reward=float(out.reward),
            kills_delta=kills_delta,
            damage_delta=damage_delta,
            nearest_zombie_distance=nearest,
            shots_delta=shots_delta,
            hits_delta=hits_delta,
        )

        self._last_info = info
        return out.observation, reward, out.terminated, out.truncated, info

    def render(self):
        return None

    def close(self):
        return None
