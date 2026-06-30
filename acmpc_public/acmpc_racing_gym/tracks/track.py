"""Track state management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from acmpc_racing_gym.tracks.gate import Gate


@dataclass
class TrackStart:
    position: np.ndarray
    yaw: float = 0.0


@dataclass
class FinishRegion:
    position: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.radius = float(self.radius)
        if self.position.shape != (3,) or not np.all(np.isfinite(self.position)):
            raise ValueError(f"Finish position must be a finite 3-vector, got {self.position}")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError(f"Finish radius must be positive, got {self.radius}")

    def contains(self, position: np.ndarray) -> bool:
        return bool(np.linalg.norm(np.asarray(position, dtype=np.float64) - self.position) <= self.radius)

    def segment_intersects(self, previous: np.ndarray, current: np.ndarray) -> bool:
        previous = np.asarray(previous, dtype=np.float64)
        current = np.asarray(current, dtype=np.float64)
        segment = current - previous
        squared_length = float(np.dot(segment, segment))
        if squared_length <= 1e-12:
            return self.contains(current)
        interpolation = float(np.clip(np.dot(self.position - previous, segment) / squared_length, 0.0, 1.0))
        closest = previous + interpolation * segment
        return bool(np.linalg.norm(closest - self.position) <= self.radius)


class Track:
    def __init__(
        self,
        name: str,
        gates: List[Gate],
        start: TrackStart,
        finish: FinishRegion,
        world_bounds: Optional[np.ndarray] = None,
    ):
        if not gates:
            raise ValueError("Track must contain at least one gate")
        self.name = name
        self.gates = gates
        self.start = start
        self.finish = finish
        self.world_bounds = world_bounds
        self.current_index = 0
        last_gate = self.gates[-1]
        self._finish_observation_gate = Gate(
            center=self.finish.position,
            normal=last_gate.normal,
            up=last_gate.up,
            width=2.0 * self.finish.radius,
            height=2.0 * self.finish.radius,
            frame_thickness=0.0,
            label="FINISH",
        )

    def reset(self) -> None:
        self.current_index = 0

    def current_gate(self) -> Gate:
        if self.in_finish_phase():
            raise RuntimeError("No current race gate remains; the track is in its finish phase")
        return self.gates[min(self.current_index, len(self.gates) - 1)]

    def future_gates(self, count: int = 2) -> List[Gate]:
        gates = []
        for offset in range(count):
            idx = self.current_index + offset
            gates.append(self.gates[idx] if idx < len(self.gates) else self._finish_observation_gate)
        return gates

    def advance_gate(self) -> None:
        if self.current_index < len(self.gates):
            self.current_index += 1

    def in_finish_phase(self) -> bool:
        return self.current_index >= len(self.gates)

    def target_position(self) -> np.ndarray:
        if self.in_finish_phase():
            return self.finish.position
        return self.current_gate().center

    def finish_reached(self, position: np.ndarray) -> bool:
        return self.in_finish_phase() and self.finish.contains(position)

    def finish_crossed(self, previous: np.ndarray, current: np.ndarray) -> bool:
        return self.in_finish_phase() and self.finish.segment_intersects(previous, current)
