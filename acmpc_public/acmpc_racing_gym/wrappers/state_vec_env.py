"""Vectorized environment wrapper exposing AC-MPC physical state."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv


class StateDummyVecEnv(DummyVecEnv):
    """DummyVecEnv with a get_state() method for AC-MPC policies."""

    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.state_space = self.envs[0].state_space

    def get_state(self) -> np.ndarray:
        states = self.env_method("get_state")
        return np.asarray(states, dtype=np.float32).reshape(self.num_envs, -1)
