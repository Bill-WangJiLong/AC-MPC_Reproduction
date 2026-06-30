"""Smoke test for Flightmare RacingEnv_v1.

This script assumes the Flightmare `flightgym` extension has already been built
and installed in the active Python environment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test Flightmare RacingEnv_v1.")
    parser.add_argument(
        "--flightmare-path",
        type=Path,
        default=Path(r"D:\MyProjects\flightmare"),
        help="Path used to set FLIGHTMARE_PATH before importing flightgym.",
    )
    parser.add_argument("--steps", type=int, default=5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    os.environ.setdefault("FLIGHTMARE_PATH", str(args.flightmare_path))

    from flightgym import RacingEnv_v1

    env = RacingEnv_v1()
    n_envs = env.getNumOfEnvs()
    obs_dim = env.getObsDim()
    act_dim = env.getActDim()
    state_dim = env.getStateDim()
    extra_names = list(env.getExtraInfoNames())

    assert obs_dim == 36, obs_dim
    assert act_dim == 4, act_dim
    assert state_dim == 13, state_dim

    obs = np.zeros((n_envs, obs_dim), dtype=np.float32)
    state = np.zeros((n_envs, state_dim), dtype=np.float32)
    action = np.zeros((n_envs, act_dim), dtype=np.float32)
    reward = np.zeros(n_envs, dtype=np.float32)
    done = np.zeros(n_envs, dtype=np.bool_)
    extra = np.zeros((n_envs, len(extra_names)), dtype=np.float32)

    assert env.reset(obs)
    assert env.getState(state)
    assert np.all(np.isfinite(obs))
    assert np.all(np.isfinite(state))

    for _ in range(args.steps):
        assert env.step(action, obs, reward, done, extra)
        assert env.getState(state)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(state))
        assert np.all(np.isfinite(reward))

    print("RacingEnv_v1 smoke test passed")
    print("n_envs:", n_envs)
    print("obs_dim:", obs_dim)
    print("act_dim:", act_dim)
    print("state_dim:", state_dim)
    print("extra_info:", extra_names)
    print("last_reward:", reward.tolist())
    print("last_done:", done.tolist())


if __name__ == "__main__":
    main()
