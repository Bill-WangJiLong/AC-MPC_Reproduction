"""AC-MPC paper-style racing reward."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardConfig:
    collision_reward: float = -10.0
    gate_pass_reward: float = 10.0
    finish_reward: float = 10.0
    body_rate_coeff: float = 0.01


class RacingReward:
    def __init__(self, config: RewardConfig):
        self.config = config

    def compute(
        self,
        prev_position: np.ndarray,
        curr_position: np.ndarray,
        target_gate_center: np.ndarray,
        body_rate: np.ndarray,
        collision: bool,
        gate_passed: bool,
        race_finished: bool,
    ) -> float:
        if collision:
            return float(self.config.collision_reward)
        if race_finished:
            reward = self.config.finish_reward
            if gate_passed:
                reward += self.config.gate_pass_reward
            return float(reward)
        if gate_passed:
            return float(self.config.gate_pass_reward)

        prev_distance = np.linalg.norm(target_gate_center - prev_position)
        curr_distance = np.linalg.norm(target_gate_center - curr_position)
        progress = prev_distance - curr_distance
        body_rate_penalty = self.config.body_rate_coeff * np.linalg.norm(body_rate)
        return float(progress - body_rate_penalty)
