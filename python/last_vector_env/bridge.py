"""Python bridge helpers for the native `last_vector_core` module."""

from __future__ import annotations

import numpy as np

import last_vector_core


ACTION_DIM = 8


def make_simulator(seed: int = 0, episode_seconds: float = 180.0) -> last_vector_core.Simulator:
    """Create a native simulator with deterministic seed and episode limit."""

    return last_vector_core.Simulator(seed=seed, episode_seconds=episode_seconds)


def ensure_action(action: np.ndarray) -> np.ndarray:
    """Convert action to contiguous float32 vector and validate shape."""

    arr = np.ascontiguousarray(np.asarray(action, dtype=np.float32).reshape(-1))
    if arr.shape[0] != ACTION_DIM:
        raise ValueError(f"Action must contain exactly {ACTION_DIM} values, got {arr.shape[0]}.")
    return arr
