"""Configuration objects for the AC-MPC racing Gym environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from acmpc_racing_gym.dynamics.params import QuadrotorParams
from acmpc_racing_gym.observations.acmpc_observation import ObservationConfig
from acmpc_racing_gym.rewards.racing_reward import RewardConfig


Vector3Tuple = Tuple[float, float, float]
BoundsTuple = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]


@dataclass
class InitialStateConfig:
    position: Optional[Vector3Tuple] = None
    velocity: Vector3Tuple = (0.0, 0.0, 0.0)
    yaw: Optional[float] = None
    position_noise: Vector3Tuple = (0.05, 0.05, 0.02)
    velocity_noise: Vector3Tuple = (0.02, 0.02, 0.02)
    yaw_noise: float = 0.02


@dataclass
class RacingEnvConfig:
    track_name: str = "split_s"
    track_path: Optional[Path] = None
    random_reset: bool = True
    max_episode_steps: int = 500
    drone_radius: float = 0.18
    world_bounds: BoundsTuple = ((-8.0, 12.0), (-6.0, 6.0), (0.0, 7.0))
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)
    dynamics: QuadrotorParams = field(default_factory=QuadrotorParams)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    seed: Optional[int] = None

    def world_bounds_array(self) -> np.ndarray:
        return np.asarray(self.world_bounds, dtype=np.float32)
