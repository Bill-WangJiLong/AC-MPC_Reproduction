"""Quadrotor parameters for the Flightmare-like Python dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class QuadrotorParams:
    mass: float = 0.752
    gravity: float = 9.8066
    dt: float = 0.02
    substep_dt: float = 0.0025

    arm_l: float = 0.17
    inertia_diag: Tuple[float, float, float] = (0.0025, 0.0021, 0.0043)
    kappa: float = 0.016

    thrust_min_per_motor: float = 0.0
    thrust_max_per_motor: float = 8.5

    omega_cmd_max: Tuple[float, float, float] = (10.0, 10.0, 4.0)
    rate_gain: Tuple[float, float, float] = (16.6, 16.6, 5.0)

    motor_tau: float = 0.02
    motor_omega_min: float = 150.0
    motor_omega_max: float = 3000.0
    thrust_map: Tuple[float, float, float] = (
        1.3298253500372892e-06,
        0.0038360810526746033,
        -1.7689986848125325,
    )

    linear_drag: Tuple[float, float, float] = (0.05, 0.05, 0.08)

    @property
    def collective_thrust_min(self) -> float:
        return 4.0 * self.thrust_min_per_motor

    @property
    def collective_thrust_max(self) -> float:
        return 4.0 * self.thrust_max_per_motor
