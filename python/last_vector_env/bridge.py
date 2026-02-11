from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

import last_vector_core


@dataclass
class BridgeStep:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class SimulatorBridge:
    """Thin Python bridge to the real C++ deterministic simulator."""

    def __init__(self) -> None:
        self._sim = last_vector_core.Simulator()
        self.obs_dim = int(self._sim.observation_dim())
        self.action_dim = int(last_vector_core.Simulator.action_dim())

    def reset(self, seed: int) -> np.ndarray:
        obs = self._sim.reset(int(seed))
        return np.asarray(obs, dtype=np.float32)

    def step(self, action: np.ndarray) -> BridgeStep:
        clipped = np.asarray(action, dtype=np.float32).copy()
        if clipped.shape != (self.action_dim,):
            raise ValueError(f"action shape must be ({self.action_dim},)")
        clipped[:4] = np.clip(clipped[:4], -1.0, 1.0)
        clipped[4:7] = np.clip(clipped[4:7], 0.0, 1.0)
        if clipped[7] < -0.5:
            clipped[7] = -1.0
        else:
            clipped[7] = float(np.clip(np.rint(clipped[7]), 0.0, 2.0))

        obs, reward, terminated, truncated, info = self._sim.step(clipped)
        return BridgeStep(
            observation=np.asarray(obs, dtype=np.float32),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )
