"""Gym-compatible AC-MPC racing environment."""

from __future__ import annotations

from typing import Optional, Tuple

import gym
import numpy as np
from gym import spaces

from acmpc_racing_gym.config import RacingEnvConfig
from acmpc_racing_gym.dynamics.flightmare_like_dynamics import FlightmareLikeDynamics
from acmpc_racing_gym.dynamics.state import QuadrotorState, normalize_quat, yaw_to_quat
from acmpc_racing_gym.observations.acmpc_observation import AcMpcObservationBuilder
from acmpc_racing_gym.rewards.racing_reward import RacingReward
from acmpc_racing_gym.tracks.loader import load_track


class AcMpcRacingEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config: Optional[RacingEnvConfig] = None):
        super().__init__()
        self.config = config or RacingEnvConfig()
        self.np_random = np.random.default_rng(self.config.seed)
        self.track = load_track(self.config.track_name, self.config.track_path)
        self._apply_track_world_bounds()
        self.dynamics = FlightmareLikeDynamics(self.config.dynamics)
        self.observation_builder = AcMpcObservationBuilder(self.config.observation)
        self.reward_fn = RacingReward(self.config.reward)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_builder.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
        self.state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32,
        )

        self.steps = 0
        self.last_info = {}
        self.reset()

    def seed(self, seed: Optional[int] = None):
        self.config.seed = seed
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def reset(self):
        self.track = load_track(self.config.track_name, self.config.track_path)
        self._apply_track_world_bounds()
        self.track.reset()
        self.steps = 0

        state = self._sample_initial_state()
        self.dynamics.reset(state)
        obs = self._get_obs()
        self.last_info = {}
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(4)
        prev_position = self.dynamics.state.position.copy() #记录当前门
        was_finish_phase = self.track.in_finish_phase() #记录当前是否在终点阶段
        target_gate = None if was_finish_phase else self.track.current_gate() #记录目标门
        target_position = self.track.target_position().copy()

        state, command = self.dynamics.step(action)
        curr_position = state.position.copy()
        self.steps += 1

        gate_passed = False
        frame_collision = False
        if target_gate is not None:
            gate_passed = target_gate.check_pass(
                prev_position,
                curr_position,
                drone_radius=self.config.drone_radius,
                direction_required=True,
            )
            frame_collision = target_gate.check_frame_collision(
                prev_position,
                curr_position,
                drone_radius=self.config.drone_radius,
            )

        if gate_passed:
            self.track.advance_gate()
        race_finished = bool(
            self.track.in_finish_phase()
            and self.track.finish.segment_intersects(prev_position, curr_position)
        )

        out_of_bounds = self._out_of_bounds(curr_position)
        ground_collision = curr_position[2] <= self.config.world_bounds[2][0] + self.config.drone_radius
        finite_state = np.all(np.isfinite(self.get_state()))
        timeout = self.steps >= self.config.max_episode_steps

        collision = bool(frame_collision or ground_collision or out_of_bounds or not finite_state)
        done = bool(collision or race_finished or timeout)
        reward = self.reward_fn.compute(
            prev_position=prev_position,
            curr_position=curr_position,
            target_gate_center=target_position,
            body_rate=state.omega,
            collision=collision,
            gate_passed=gate_passed,
            race_finished=race_finished,
        )

        obs = self._get_obs()
        info = {
            "gate_index": int(self.track.current_index),
            "gate_passed": bool(gate_passed),
            "finish_phase": bool(self.track.in_finish_phase()),
            "finish_distance": float(np.linalg.norm(curr_position - self.track.finish.position)),
            "collision": collision,
            "collision_type": self._collision_type(
                frame_collision=frame_collision,
                ground_collision=ground_collision,
                out_of_bounds=out_of_bounds,
                finite_state=finite_state,
            ),
            "finished": bool(race_finished),
            "out_of_bounds": bool(out_of_bounds),
            "timeout": bool(timeout),
            "position": state.position.astype(np.float32).copy(),
            "velocity": state.velocity.astype(np.float32).copy(),
            "omega": state.omega.astype(np.float32).copy(),
            "physical_command": command.as_info(),
        }
        if timeout and not (collision or race_finished):
            info["TimeLimit.truncated"] = True
        self.last_info = info
        return obs, float(reward), done, info

    def get_state(self) -> np.ndarray:
        return self.dynamics.get_state13()

    def compute_prediction_rollout(self, prediction: np.ndarray) -> dict:
        """Evaluate MPC-predicted states/actions without changing env state.

        prediction shape is [H, 14] with [position(3), quaternion(4),
        velocity(3), control(4)]. The control is [collective thrust, wx, wy,
        wz]; only body rates are used by the racing reward.
        """
        prediction = np.asarray(prediction, dtype=np.float64)
        if prediction.ndim != 2 or prediction.shape[1] != 14:
            raise ValueError(f"Expected prediction shape [H, 14], got {prediction.shape}")

        horizon = prediction.shape[0]
        pred_obs = np.zeros((horizon, self.observation_builder.obs_dim), dtype=np.float32)
        pred_rewards = np.zeros(horizon, dtype=np.float32)
        pred_valid = np.zeros(horizon, dtype=np.float32)
        pred_terminal = np.zeros(horizon, dtype=np.float32)

        original_gate_index = self.track.current_index
        simulated_gate_index = int(self.track.current_index)
        prev_position = self.dynamics.state.position.copy()
        terminal_reached = False

        try:
            for step_idx in range(horizon):
                if terminal_reached:
                    break

                x_mpc = prediction[step_idx, :10]
                u_mpc = prediction[step_idx, 10:14]
                finite_prediction = bool(np.all(np.isfinite(x_mpc)) and np.all(np.isfinite(u_mpc)))
                if not finite_prediction:
                    break

                body_rate = u_mpc[1:4]
                predicted_state = QuadrotorState(
                    position=x_mpc[0:3].copy(),
                    quaternion=normalize_quat(x_mpc[3:7]),
                    velocity=x_mpc[7:10].copy(),
                    omega=body_rate.copy(),
                )
                curr_position = predicted_state.position.copy()

                self.track.current_index = simulated_gate_index
                was_finish_phase = simulated_gate_index >= len(self.track.gates)
                target_gate = None if was_finish_phase else self.track.current_gate()
                target_position = self.track.target_position().copy()
                gate_passed = False
                frame_collision = False
                if target_gate is not None:
                    gate_passed = target_gate.check_pass(
                        prev_position,
                        curr_position,
                        drone_radius=self.config.drone_radius,
                        direction_required=True,
                    )
                    frame_collision = target_gate.check_frame_collision(
                        prev_position,
                        curr_position,
                        drone_radius=self.config.drone_radius,
                    )

                if gate_passed:
                    simulated_gate_index += 1
                race_finished = bool(
                    simulated_gate_index >= len(self.track.gates)
                    and self.track.finish.segment_intersects(prev_position, curr_position)
                )

                out_of_bounds = self._out_of_bounds(curr_position)
                ground_collision = curr_position[2] <= self.config.world_bounds[2][0] + self.config.drone_radius
                collision = bool(frame_collision or ground_collision or out_of_bounds)

                self.track.current_index = simulated_gate_index
                pred_obs[step_idx] = self.observation_builder.build(predicted_state, self.track)
                pred_rewards[step_idx] = self.reward_fn.compute(
                    prev_position=prev_position,
                    curr_position=curr_position,
                    target_gate_center=target_position,
                    body_rate=body_rate,
                    collision=collision,
                    gate_passed=gate_passed,
                    race_finished=race_finished,
                )
                pred_valid[step_idx] = 1.0

                if collision or race_finished:
                    pred_terminal[step_idx] = 1.0
                    terminal_reached = True

                prev_position = curr_position
        finally:
            self.track.current_index = original_gate_index

        return {
            "observations": pred_obs,
            "rewards": pred_rewards,
            "valid": pred_valid,
            "terminal": pred_terminal,
        }

    def render(self, mode="human"):
        if mode == "human":
            pos = self.dynamics.state.position
            print(f"step={self.steps} pos={pos} gate={self.track.current_index}")
            return None
        if mode == "rgb_array":
            return np.zeros((64, 64, 3), dtype=np.uint8)
        raise NotImplementedError(f"Unsupported render mode: {mode}")

    def close(self):
        return None

    def _get_obs(self) -> np.ndarray:
        return self.observation_builder.build(self.dynamics.state, self.track)

    def _sample_initial_state(self) -> QuadrotorState:
        start = self.track.start
        position = start.position.copy()
        velocity = np.zeros(3, dtype=np.float64)
        yaw = float(start.yaw)

        if self.config.random_reset:
            init = self.config.initial_state
            if init.position is not None:
                position = np.asarray(init.position, dtype=np.float64)
            position += self.np_random.uniform(-np.asarray(init.position_noise), np.asarray(init.position_noise))
            velocity = np.asarray(init.velocity, dtype=np.float64)
            velocity += self.np_random.uniform(-np.asarray(init.velocity_noise), np.asarray(init.velocity_noise))
            if init.yaw is not None:
                yaw = float(init.yaw)
            yaw += float(self.np_random.uniform(-init.yaw_noise, init.yaw_noise))
        return QuadrotorState(
            position=position,
            quaternion=yaw_to_quat(yaw),
            velocity=velocity,
            omega=np.zeros(3, dtype=np.float64),
        )

    def _out_of_bounds(self, position: np.ndarray) -> bool:
        bounds = self.config.world_bounds_array()
        return bool(np.any(position < bounds[:, 0]) or np.any(position > bounds[:, 1]))

    def _apply_track_world_bounds(self) -> None:
        if self.track.world_bounds is None:
            return
        bounds = np.asarray(self.track.world_bounds, dtype=np.float32)
        self.config.world_bounds = tuple((float(low), float(high)) for low, high in bounds)

    @staticmethod
    def _collision_type(
        frame_collision: bool,
        ground_collision: bool,
        out_of_bounds: bool,
        finite_state: bool,
    ) -> Optional[str]:
        if not finite_state:
            return "non_finite_state"
        if out_of_bounds:
            return "out_of_bounds"
        if ground_collision:
            return "ground"
        if frame_collision:
            return "gate_frame"
        return None
