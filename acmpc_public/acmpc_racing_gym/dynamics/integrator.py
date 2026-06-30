"""Numerical integration helpers."""

from __future__ import annotations

from typing import Callable

import numpy as np


DerivativeFn = Callable[[np.ndarray], np.ndarray]


def rk4_step(f: DerivativeFn, state: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
