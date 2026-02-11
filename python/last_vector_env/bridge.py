"""Python bridge helpers for the native `last_vector_core` module."""

from __future__ import annotations

import numpy as np

import last_vector_core


def make_simulator(seed: int = 0, episode_seconds: float = 180.0) -> last_vector_core.Simulator:
    return last_vector_core.Simulator(seed=seed, episode_seconds=episode_seconds)


def ensure_action(action: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(action, dtype=np.float32).reshape(-1))
