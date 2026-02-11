from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .bridge import MockDeterministicCore


@dataclass
class EnvConfig:
    episode_limit_s: float = 180.0
    fps: int = 60


class LastVectorEnv(gym.Env[np.ndarray, Dict[str, np.ndarray]]):
    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(self, config: Optional[EnvConfig] = None, render_mode: str = "none") -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.core = MockDeterministicCore()

        self.action_space = spaces.Dict(
            {
                "continuous": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "binary": spaces.MultiBinary(3),
                "upgrade_choice": spaces.Discrete(4),  # 0=no-op, 1..3 choose card index
            }
        )

        obs_len = 11 + 8 * 5 + 16 + 2 + 8
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_len,), dtype=np.float32)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        s = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        obs = np.asarray(self.core.reset(s), dtype=np.float32)
        return obs, {"seed": s}

    def step(self, action: Dict[str, np.ndarray]):
        cont = action["continuous"].astype(np.float32)
        binary = action["binary"]
        upgrade = int(action["upgrade_choice"])

        cxx_action = {
            "move_x": float(cont[0]),
            "move_y": float(cont[1]),
            "aim_x": float(cont[2]),
            "aim_y": float(cont[3]),
            "shoot": float(binary[0]),
            "sprint": float(binary[1]),
            "reload": float(binary[2]),
            "upgrade_choice": upgrade - 1,
        }
        out = self.core.step(cxx_action)
        obs = np.asarray(out.observation, dtype=np.float32)
        info = dict(out.info)
        return obs, float(out.reward), bool(out.terminated), bool(out.truncated), info

    def render(self):
        return None

    def close(self):
        return None
