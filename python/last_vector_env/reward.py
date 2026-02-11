from __future__ import annotations


def shaped_reward(
    base_reward: float,
    *,
    kills_delta: float,
    damage_delta: float,
    nearest_zombie_distance: float,
    shots_delta: float,
    hits_delta: float,
    damage_dealt_delta: float = 0.0,
) -> float:
    """Reward helper mirroring native simulator shaping.

    Kill reward tuned up by ~16% (1.25 -> 1.45) to nudge combat behavior.
    """

    reward = base_reward
    reward += 1.45 * kills_delta
    reward += 0.03 * hits_delta
    reward += 0.002 * damage_dealt_delta
    reward -= 0.05 * damage_delta
    if nearest_zombie_distance < 120.0:
        reward -= (120.0 - nearest_zombie_distance) * 0.0008
    if shots_delta > 0 and hits_delta <= 0:
        reward -= 0.008 * shots_delta
    return reward
