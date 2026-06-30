"""Gate geometry and crossing checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm < 1e-12 or not np.isfinite(norm):
        raise ValueError(f"Cannot normalize vector {vector}")
    return vector / norm


@dataclass
class Gate:
    center: np.ndarray
    normal: np.ndarray
    up: np.ndarray
    width: float
    height: float
    frame_thickness: float = 0.12
    label: Optional[str] = None

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64)
        self.normal = _normalize(self.normal)
        self.up = _normalize(self.up - np.dot(self.up, self.normal) * self.normal)
        self.right = _normalize(np.cross(self.up, self.normal))

    def corners_world(self) -> np.ndarray:
        half_w = 0.5 * self.width
        half_h = 0.5 * self.height
        return np.asarray(
            [
                self.center - half_w * self.right - half_h * self.up,
                self.center + half_w * self.right - half_h * self.up,
                self.center + half_w * self.right + half_h * self.up,
                self.center - half_w * self.right + half_h * self.up,
            ],
            dtype=np.float64,
        )

    def world_to_gate(self, point: np.ndarray) -> np.ndarray:
        delta = np.asarray(point, dtype=np.float64) - self.center
        return np.array(
            [
                np.dot(delta, self.right),
                np.dot(delta, self.up),
                np.dot(delta, self.normal),
            ],
            dtype=np.float64,
        )

    def segment_plane_intersection(
        self,
        p_prev: np.ndarray,
        p_curr: np.ndarray,
        direction_required: bool = False,
    ) -> Optional[Tuple[np.ndarray, float]]:
        p_prev = np.asarray(p_prev, dtype=np.float64)
        p_curr = np.asarray(p_curr, dtype=np.float64)
        d0 = float(np.dot(p_prev - self.center, self.normal))
        d1 = float(np.dot(p_curr - self.center, self.normal))

        if direction_required:
            if not (d0 < 0.0 <= d1):
                return None
        else:
            if d0 == 0.0 and d1 == 0.0:
                return None
            if d0 * d1 > 0.0:
                return None

        denom = d1 - d0
        if abs(denom) < 1e-12:
            return None
        t = -d0 / denom
        if t < -1e-9 or t > 1.0 + 1e-9:
            return None
        t = float(np.clip(t, 0.0, 1.0))
        point = p_prev + t * (p_curr - p_prev)
        return point, t

    def check_pass(
        self,
        p_prev: np.ndarray,
        p_curr: np.ndarray,
        drone_radius: float = 0.0,
        direction_required: bool = True,
    ) -> bool:
        hit = self.segment_plane_intersection(p_prev, p_curr, direction_required=direction_required)
        if hit is None:
            return False
        point, _ = hit
        local = self.world_to_gate(point)
        half_w = max(0.0, 0.5 * self.width - drone_radius)
        half_h = max(0.0, 0.5 * self.height - drone_radius)
        return abs(local[0]) <= half_w and abs(local[1]) <= half_h

    def check_frame_collision(
        self,
        p_prev: np.ndarray,
        p_curr: np.ndarray,
        drone_radius: float,
    ) -> bool:
        hit = self.segment_plane_intersection(p_prev, p_curr, direction_required=False)
        if hit is None:
            return False
        point, _ = hit
        local = self.world_to_gate(point)

        inner_w = max(0.0, 0.5 * self.width - drone_radius)
        inner_h = max(0.0, 0.5 * self.height - drone_radius)
        outer_w = 0.5 * self.width + self.frame_thickness + drone_radius
        outer_h = 0.5 * self.height + self.frame_thickness + drone_radius

        inside_inner = abs(local[0]) <= inner_w and abs(local[1]) <= inner_h
        inside_outer = abs(local[0]) <= outer_w and abs(local[1]) <= outer_h
        return inside_outer and not inside_inner
