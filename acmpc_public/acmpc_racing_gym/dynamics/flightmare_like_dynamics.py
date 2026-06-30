"""Flightmare-like rigid-body quadrotor dynamics in Python."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from acmpc_racing_gym.dynamics.integrator import rk4_step
from acmpc_racing_gym.dynamics.params import QuadrotorParams
from acmpc_racing_gym.dynamics.state import (
    QuadrotorState,
    normalize_quat,
    quat_multiply,
    quat_to_rotmat,
)


@dataclass
class PhysicalCommand:
    mass_normalized_thrust: float
    collective_thrust_n: float
    body_rate_cmd: np.ndarray
    normalized_action: np.ndarray

    def as_info(self) -> Dict[str, object]:
        return {
            "mass_normalized_thrust": float(self.mass_normalized_thrust),
            "collective_thrust_N": float(self.collective_thrust_n),
            "body_rate_cmd": self.body_rate_cmd.astype(np.float32).copy(),
        }


class FlightmareLikeDynamics:
    """Rigid-body quadrotor model with rate control, motor lag and RK4."""

    def __init__(self, params: QuadrotorParams):
        self.params = params
        self.inertia = np.diag(np.asarray(params.inertia_diag, dtype=np.float64))
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.rate_gain = np.diag(np.asarray(params.rate_gain, dtype=np.float64))
        self.omega_cmd_max = np.asarray(params.omega_cmd_max, dtype=np.float64)
        self.linear_drag = np.asarray(params.linear_drag, dtype=np.float64)
        self.allocation = self._build_allocation_matrix()
        self.allocation_inv = np.linalg.inv(self.allocation)
        self.state = QuadrotorState.zero()
        self.motor_omega = np.zeros(4, dtype=np.float64)
        self.last_command = PhysicalCommand(0.0, 0.0, np.zeros(3), np.zeros(4))

    def reset(self, state: QuadrotorState | None = None) -> QuadrotorState:
        self.state = state.copy() if state is not None else QuadrotorState.zero()
        self.state.quaternion = normalize_quat(self.state.quaternion)
        hover_thrust = self.params.mass * self.params.gravity / 4.0
        self.motor_omega = self._thrust_to_motor_omega(np.full(4, hover_thrust))
        self.last_command = PhysicalCommand(
            mass_normalized_thrust=self.params.gravity,
            collective_thrust_n=self.params.mass * self.params.gravity,
            body_rate_cmd=np.zeros(3, dtype=np.float64),
            normalized_action=np.zeros(4, dtype=np.float64),
        )
        return self.state.copy()

    def get_state13(self) -> np.ndarray:
        return self.state.as_vector13()

    def action_to_command(self, action: np.ndarray) -> PhysicalCommand:
        action = np.asarray(action, dtype=np.float64).reshape(4)
        action = np.clip(action, -1.0, 1.0)

        normalization_max = self.params.thrust_max_per_motor
        force_mean = (normalization_max * 4.0 / self.params.mass) / 2.0
        force_std = force_mean
        mass_normalized_thrust = action[0] * force_std + force_mean
        collective_thrust_n = self.params.mass * mass_normalized_thrust
        collective_thrust_n = float(
            np.clip(
                collective_thrust_n,
                self.params.collective_thrust_min,
                self.params.collective_thrust_max,
            )
        )
        mass_normalized_thrust = collective_thrust_n / self.params.mass

        body_rate_cmd = action[1:4] * self.omega_cmd_max
        body_rate_cmd = np.clip(body_rate_cmd, -self.omega_cmd_max, self.omega_cmd_max)
        return PhysicalCommand(
            mass_normalized_thrust=float(mass_normalized_thrust),
            collective_thrust_n=collective_thrust_n,
            body_rate_cmd=body_rate_cmd,
            normalized_action=action.copy(),
        )

    def step(self, action: np.ndarray) -> Tuple[QuadrotorState, PhysicalCommand]:
        command = self.action_to_command(action)
        remaining = self.params.dt

        while remaining > 1e-12:
            dt = min(self.params.substep_dt, remaining)
            self._substep(command, dt)
            remaining -= dt

        self.last_command = command
        return self.state.copy(), command

    def _substep(self, command: PhysicalCommand, dt: float) -> None:
        thrusts_des = self._desired_motor_thrusts(command)
        thrusts_des = self._clamp_motor_thrust(thrusts_des)

        motor_omega_des = self._thrust_to_motor_omega(thrusts_des)
        if self.params.motor_tau <= 0.0:
            self.motor_omega = motor_omega_des
        else:
            coeff = np.exp(-dt / self.params.motor_tau)
            self.motor_omega = coeff * self.motor_omega + (1.0 - coeff) * motor_omega_des
        self.motor_omega = np.clip(
            self.motor_omega,
            self.params.motor_omega_min,
            self.params.motor_omega_max,
        )

        motor_thrusts = self._clamp_motor_thrust(self._motor_omega_to_thrust(self.motor_omega))
        wrench = self.allocation @ motor_thrusts
        force = float(np.clip(wrench[0], self.params.collective_thrust_min, self.params.collective_thrust_max))
        torque = wrench[1:4]

        state_vec = self.state.as_dynamics_vector()

        def derivative(vec: np.ndarray) -> np.ndarray:
            return self._derivative(vec, force, torque)

        next_vec = rk4_step(derivative, state_vec, dt)
        next_vec[3:7] = normalize_quat(next_vec[3:7])
        self.state = QuadrotorState.from_dynamics_vector(next_vec)

    def _desired_motor_thrusts(self, command: PhysicalCommand) -> np.ndarray:
        omega = self.state.omega
        omega_error = command.body_rate_cmd - omega
        torque_des = self.inertia @ self.rate_gain @ omega_error + np.cross(omega, self.inertia @ omega)
        wrench_des = np.concatenate([[command.collective_thrust_n], torque_des])
        return self.allocation_inv @ wrench_des

    def _derivative(self, vec: np.ndarray, force: float, torque: np.ndarray) -> np.ndarray:
        position = vec[0:3]
        del position
        quat = normalize_quat(vec[3:7])
        velocity = vec[7:10]
        omega = vec[10:13]

        dvec = np.zeros_like(vec)
        dvec[0:3] = velocity

        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=np.float64)
        dvec[3:7] = 0.5 * quat_multiply(quat, omega_quat)

        thrust_world = quat_to_rotmat(quat) @ np.array([0.0, 0.0, force], dtype=np.float64)
        gravity = np.array([0.0, 0.0, -self.params.gravity], dtype=np.float64)
        drag_acc = -(self.linear_drag * velocity) / self.params.mass
        dvec[7:10] = thrust_world / self.params.mass + gravity + drag_acc

        dvec[10:13] = self.inertia_inv @ (torque - np.cross(omega, self.inertia @ omega))
        return dvec

    def _build_allocation_matrix(self) -> np.ndarray:
        arm = self.params.arm_l * np.sqrt(0.5)
        return np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [arm, -arm, -arm, arm],
                [-arm, -arm, arm, arm],
                [
                    self.params.kappa,
                    -self.params.kappa,
                    self.params.kappa,
                    -self.params.kappa,
                ],
            ],
            dtype=np.float64,
        )

    def _motor_omega_to_thrust(self, omega: np.ndarray) -> np.ndarray:
        a, b, c = self.params.thrust_map
        thrust = a * omega * omega + b * omega + c
        return self._clamp_motor_thrust(thrust)

    def _thrust_to_motor_omega(self, thrusts: np.ndarray) -> np.ndarray:
        thrusts = self._clamp_motor_thrust(np.asarray(thrusts, dtype=np.float64))
        a, b, c = self.params.thrust_map
        if abs(a) < 1e-12:
            omega = (thrusts - c) / max(b, 1e-12)
        else:
            disc = b * b - 4.0 * a * (c - thrusts)
            disc = np.maximum(disc, 0.0)
            omega = (-b + np.sqrt(disc)) / (2.0 * a)
        return np.clip(omega, self.params.motor_omega_min, self.params.motor_omega_max)

    def _clamp_motor_thrust(self, thrusts: np.ndarray) -> np.ndarray:
        return np.clip(
            thrusts,
            self.params.thrust_min_per_motor,
            self.params.thrust_max_per_motor,
        )
