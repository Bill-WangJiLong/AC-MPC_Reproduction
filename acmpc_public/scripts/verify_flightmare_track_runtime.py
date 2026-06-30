"""Verify that Flightmare C++ RacingEnv actually loaded the installed track."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the active Flightmare RacingEnv track.")
    parser.add_argument("--flightmare-path", type=Path, default=Path(r"D:\MyProjects\flightmare"))
    parser.add_argument("--expected-track-name", default="split_s")
    parser.add_argument("--expected-gates", type=int, default=7)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    flightmare_path = args.flightmare_path.resolve()
    os.environ["FLIGHTMARE_PATH"] = str(flightmare_path)

    from acmpc_flightmare.track import load_flightmare_track
    from acmpc_flightmare.vec_env import ensure_flightgym_importable

    ensure_flightgym_importable(flightmare_path)
    from flightgym import RacingEnv_v1

    racing_config = flightmare_path / "flightlib" / "configs" / "racing_env.yaml"
    track = load_flightmare_track(racing_config)
    if track["name"] != args.expected_track_name:
        raise AssertionError(f"Expected track {args.expected_track_name!r}, got {track['name']!r}")
    if len(track["gates"]) != args.expected_gates:
        raise AssertionError(f"Expected {args.expected_gates} gates, got {len(track['gates'])}")
    finish = track.get("finish")
    if not isinstance(finish, dict) or float(finish.get("radius", 0.0)) <= 0.0:
        raise AssertionError("Track must define a positive-radius finish region")

    env = RacingEnv_v1()
    n_envs = int(env.getNumOfEnvs())
    obs_dim = int(env.getObsDim())
    act_dim = int(env.getActDim())
    state_dim = int(env.getStateDim())
    extra_names = list(env.getExtraInfoNames())

    obs = np.zeros((n_envs, obs_dim), dtype=np.float32)
    state = np.zeros((n_envs, state_dim), dtype=np.float32)
    action = np.zeros((n_envs, act_dim), dtype=np.float32)
    reward = np.zeros(n_envs, dtype=np.float32)
    done = np.zeros(n_envs, dtype=np.bool_)
    extra = np.zeros((n_envs, len(extra_names)), dtype=np.float32)

    try:
        if not env.reset(obs):
            raise RuntimeError("Flightmare reset() failed")
        if not env.getState(state):
            raise RuntimeError("Flightmare getState() failed")

        start = np.asarray(track["start"]["position"], dtype=np.float32)
        position_error = np.abs(state[:, 0:3] - start[None, :])
        reset_tolerance = np.asarray([0.051, 0.051, 0.021], dtype=np.float32)
        if np.any(position_error > reset_tolerance[None, :] + 1e-4):
            raise AssertionError(
                f"Runtime reset position does not match track start; max error={position_error.max(axis=0)}"
            )

        gate_corners = np.asarray(track["gates"][0]["corners"], dtype=np.float32)
        expected_relative = (gate_corners - state[0, 0:3]).reshape(-1)
        runtime_relative = obs[0, 12:24]
        corner_error = float(np.max(np.abs(expected_relative - runtime_relative)))
        if corner_error > 1e-4:
            raise AssertionError(
                "Flightmare observation does not contain the expected first track gate corners; "
                f"max error={corner_error}"
            )

        if not env.step(action, obs, reward, done, extra):
            raise RuntimeError("Flightmare step() failed")
        if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(reward)):
            raise AssertionError("Flightmare returned a non-finite observation or reward")
    finally:
        env.close()

    print("Flightmare runtime track verification passed")
    print(f"track_name: {track['name']}")
    print(f"gate_count: {len(track['gates'])}")
    print(f"start_position: {track['start']['position']}")
    print(f"first_gate_center: {track['gates'][0]['center']}")
    print(f"finish_position: {finish['position']}")
    print(f"finish_radius: {finish['radius']}")
    print(f"first_gate_corner_max_error: {corner_error:.3e}")
    print(f"n_envs: {n_envs}")
    print(f"obs_dim: {obs_dim}")
    print(f"act_dim: {act_dim}")
    print(f"state_dim: {state_dim}")


if __name__ == "__main__":
    main()
