"""AC-MPC paper-style 36-dimensional observation builder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acmpc_racing_gym.dynamics.state import QuadrotorState, quat_to_rotmat
from acmpc_racing_gym.tracks.track import Track


class TrackObservationMode:
    VEHICLE_RELATIVE = "vehicle_relative"
    CHAINED_GATE_RELATIVE = "chained_gate_relative"


@dataclass
class ObservationConfig:
    future_gate_count: int = 2
    track_obs_mode: str = TrackObservationMode.VEHICLE_RELATIVE
    # Applied by the SB3 VecNormalize wrapper created in make_acmpc_racing_vec_env().
    normalize: bool = True


class AcMpcObservationBuilder:
    def __init__(self, config: ObservationConfig):
        if config.future_gate_count != 2:
            raise ValueError("AC-MPC racing observation expects exactly 2 future gates")
        self.config = config

    @property
    def obs_dim(self) -> int:
        return 36

    def build(self, drone_state: QuadrotorState, track: Track) -> np.ndarray:
        velocity = drone_state.velocity.astype(np.float64)
        rotmat = quat_to_rotmat(drone_state.quaternion).reshape(-1)

        gate1, gate2 = track.future_gates(self.config.future_gate_count)
        gate1_corners = gate1.corners_world()
        gate2_corners = gate2.corners_world()

        if self.config.track_obs_mode == TrackObservationMode.VEHICLE_RELATIVE:
            gate1_obs = gate1_corners - drone_state.position.reshape(1, 3)
            gate2_obs = gate2_corners - drone_state.position.reshape(1, 3)
        elif self.config.track_obs_mode == TrackObservationMode.CHAINED_GATE_RELATIVE:
            gate1_obs = gate1_corners - drone_state.position.reshape(1, 3)
            gate2_obs = gate2_corners - gate1_corners
        else:
            raise ValueError(f"Unknown track_obs_mode: {self.config.track_obs_mode}")

        obs = np.concatenate([velocity, rotmat, gate1_obs.reshape(-1), gate2_obs.reshape(-1)])
        if obs.shape != (self.obs_dim,):
            raise RuntimeError(f"Observation shape mismatch: expected {(self.obs_dim,)}, got {obs.shape}")
        if not np.all(np.isfinite(obs)):
            raise RuntimeError(f"Observation contains non-finite values: {obs}")
        return obs.astype(np.float32)
