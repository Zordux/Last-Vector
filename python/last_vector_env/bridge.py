"""Pure-Python deterministic bridge mirroring the C++ API contract.

Replace this module with pybind11 bindings to `lastvector_core` for production training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Tuple


@dataclass
class StepResult:
    observation: List[float]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, float]


class MockDeterministicCore:
    def __init__(self) -> None:
        self.rng = random.Random(0)
        self.t = 0
        self.hp = 1.0
        self.z = 0.0
        self.choosing = False

    def reset(self, seed: int) -> List[float]:
        self.rng.seed(seed)
        self.t = 0
        self.hp = 1.0
        self.z = 0.0
        self.choosing = False
        return self._obs()

    def step(self, action: Dict[str, float]) -> StepResult:
        self.t += 1
        if self.t % (20 * 60) == 0:
            self.choosing = True
        if self.choosing and int(action.get("upgrade_choice", -1)) in (0, 1, 2):
            self.choosing = False

        self.z += 0.01
        if action.get("shoot", 0.0) > 0.5 and self.rng.random() < 0.35:
            reward = 1.25
        else:
            reward = 0.02

        self.hp -= max(0.0, 0.003 * (1.0 + self.z) - 0.001 * action.get("sprint", 0.0))
        terminated = self.hp <= 0.0
        truncated = self.t >= 180 * 60
        reward -= (1.0 - self.hp) * 0.01

        info = {
            "kills": float(self.t // 120),
            "damage_taken": float((1.0 - max(self.hp, 0.0)) * 100.0),
            "difficulty": self.z,
            "choosing_upgrade": float(self.choosing),
        }
        return StepResult(self._obs(), reward, terminated, truncated, info)

    def _obs(self) -> List[float]:
        obs = [0.5, 0.5, 0.0, 0.0, self.hp, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        obs.extend([0.0] * (8 * 5))
        obs.extend([1.0] * 16)
        obs.extend([self.z, 1.0 if self.choosing else 0.0])
        obs.extend([0.0] * 8)
        return obs
