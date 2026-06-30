"""State representation and quaternion helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_to_quat(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


@dataclass
class QuadrotorState:
    position: np.ndarray
    quaternion: np.ndarray
    velocity: np.ndarray
    omega: np.ndarray

    @classmethod
    def zero(cls) -> "QuadrotorState":
        return cls(
            position=np.zeros(3, dtype=np.float64),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            omega=np.zeros(3, dtype=np.float64),
        )

    @classmethod
    def from_vector13(cls, vector: np.ndarray) -> "QuadrotorState":
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (13,):
            raise ValueError(f"Expected state shape (13,), got {vector.shape}")
        return cls(
            position=vector[0:3].copy(),
            quaternion=normalize_quat(vector[3:7]),
            velocity=vector[7:10].copy(),
            omega=vector[10:13].copy(),
        )

    def as_vector13(self, dtype=np.float32) -> np.ndarray:
        return np.concatenate(
            [
                self.position,
                normalize_quat(self.quaternion),
                self.velocity,
                self.omega,
            ]
        ).astype(dtype)

    def as_dynamics_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.position,
                normalize_quat(self.quaternion),
                self.velocity,
                self.omega,
            ]
        ).astype(np.float64)

    @classmethod
    def from_dynamics_vector(cls, vector: np.ndarray) -> "QuadrotorState":
        return cls.from_vector13(vector)

    def copy(self) -> "QuadrotorState":
        return QuadrotorState(
            position=self.position.copy(),
            quaternion=self.quaternion.copy(),
            velocity=self.velocity.copy(),
            omega=self.omega.copy(),
        )
