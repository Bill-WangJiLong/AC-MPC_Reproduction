"""Smoke test for the public AC-MPC core modules.

This script intentionally avoids any racing environment. It verifies that the
installed dependencies, differentiable drone model, MPC wrapper, and AC-MPC
policy class can be imported and initialized, and that a single MPC solve
returns finite tensors.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from gym import spaces


REPO_ROOT = Path(__file__).resolve().parents[1]
DIFF_MPC_DRONES = REPO_ROOT / "diff_mpc_drones"

for path in (REPO_ROOT, DIFF_MPC_DRONES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("ACMPC_T", "2")


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains non-finite values:\n{tensor}")


def smoke_imports() -> None:
    import mpc  # noqa: F401
    import drone  # noqa: F401
    import il_env  # noqa: F401
    from training_modules.mlp_mpc_policy import MlpMpcPolicy  # noqa: F401

    print("imports: ok")


def smoke_drone_forward() -> None:
    import drone

    dx = drone.DroneDx(device="cpu")
    x = torch.zeros(1, dx.n_state)
    x[:, 3] = 1.0
    u = torch.zeros(1, dx.n_ctrl)
    u[:, 0] = dx.mass * 9.8066

    x_next = dx.forward(x, u)
    assert x_next.shape == (1, dx.n_state)
    assert_finite("DroneDx.forward output", x_next)
    print("DroneDx.forward: ok", tuple(x_next.shape))


def smoke_mpc_solve() -> None:
    import drone
    import il_env

    horizon = int(os.environ["ACMPC_T"])
    batch_size = 1
    dx = drone.DroneDx(device="cpu")
    env = il_env.IL_Env("drone", mpc_T=horizon, lqr_iter=5)

    xinit = torch.zeros(batch_size, dx.n_state)
    xinit[:, 3] = 1.0

    n_tau = dx.n_state + dx.n_ctrl
    q_diag = torch.ones(horizon, batch_size, n_tau) * 0.1
    q_diag[:, :, 10] = 0.01
    q_diag[:, :, 11:] = 0.01
    q_mat = torch.diag_embed(q_diag)
    p_vec = torch.zeros(horizon, batch_size, n_tau)

    x_mpc, u_mpc = env.mpc(dx, xinit, q_mat, p_vec, lqr_iter_override=1)

    assert x_mpc.shape == (horizon, batch_size, dx.n_state)
    assert u_mpc.shape == (horizon, batch_size, dx.n_ctrl)
    assert_finite("x_mpc", x_mpc)
    assert_finite("u_mpc", u_mpc)
    print("IL_Env.mpc: ok", tuple(x_mpc.shape), tuple(u_mpc.shape))


def smoke_policy_init() -> None:
    from training_modules.mlp_mpc_policy import MlpMpcPolicy

    observation_dim = 36
    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(observation_dim,),
        dtype=np.float32,
    )
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(4,),
        dtype=np.float32,
    )

    policy = MlpMpcPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 1e-3,
    )

    assert policy.mlp_extractor.T == int(os.environ["ACMPC_T"])
    print("MlpMpcPolicy init: ok", "ACMPC_T=", policy.mlp_extractor.T)


def main() -> None:
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("torch cuda available:", torch.cuda.is_available())
    print("ACMPC_T:", os.environ["ACMPC_T"])

    smoke_imports()
    smoke_drone_forward()
    smoke_mpc_solve()
    smoke_policy_init()

    print("AC-MPC core smoke test: passed")


if __name__ == "__main__":
    main()
