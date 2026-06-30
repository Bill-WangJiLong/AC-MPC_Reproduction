"""Core AC-MPC validation checks.

This script is the Phase 1 validation target. It verifies:

- DroneDx.forward() produces finite states.
- Quaternion integration preserves unit norm for representative inputs.
- Analytic dynamics Jacobians have the expected shape and finite values.
- IL_Env.mpc() returns finite trajectories and actions inside MPC bounds.
- MlpMpcPolicy's AC-MPC actor forward works for batch sizes 1, 8, and 64.
- MlpMpcPolicy.mlp_extractor.predictions has the expected shape.
"""

from __future__ import annotations

import contextlib
import io
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


def assert_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    actual = tuple(tensor.shape)
    if actual != expected:
        raise RuntimeError(f"{name} shape mismatch: expected {expected}, got {actual}")


def assert_close(name: str, value: torch.Tensor, expected: torch.Tensor, atol: float) -> None:
    if not torch.allclose(value, expected, atol=atol, rtol=0.0):
        raise RuntimeError(f"{name} mismatch: expected {expected}, got {value}")


def validate_drone_forward_and_quaternion_norm() -> None:
    import drone

    dx = drone.DroneDx(device="cpu")
    batch_size = 4
    x = torch.zeros(batch_size, dx.n_state)
    x[:, 3] = 1.0

    u = torch.zeros(batch_size, dx.n_ctrl)
    u[:, 0] = dx.mass * 9.8066
    u[:, 1:] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, -0.1, 0.05],
            [-0.4, 0.3, -0.2],
            [0.8, -0.6, 0.25],
        ],
        dtype=torch.float32,
    )

    x_next = dx.forward(x, u)
    assert_shape("DroneDx.forward output", x_next, (batch_size, dx.n_state))
    assert_finite("DroneDx.forward output", x_next)

    quat_norm = torch.linalg.norm(x_next[:, 3:7], dim=1)
    assert_close(
        "quaternion norm after DroneDx.forward",
        quat_norm,
        torch.ones_like(quat_norm),
        atol=1e-4,
    )

    print("DroneDx.forward and quaternion norm: ok")


def validate_analytic_jacobians() -> None:
    import drone

    dx = drone.DroneDx(device="cpu")
    batch_size = 8
    x = torch.zeros(batch_size, dx.n_state)
    x[:, 3] = 1.0

    u = torch.zeros(batch_size, dx.n_ctrl)
    u[:, 0] = dx.mass * 9.8066
    u[:, 1:] = torch.randn(batch_size, 3) * 0.1

    a_mat, b_mat = dx.grad_input(x, u)
    assert_shape("A jacobian", a_mat, (batch_size, dx.n_state, dx.n_state))
    assert_shape("B jacobian", b_mat, (batch_size, dx.n_state, dx.n_ctrl))
    assert_finite("A jacobian", a_mat)
    assert_finite("B jacobian", b_mat)

    print("analytic jacobians: ok", tuple(a_mat.shape), tuple(b_mat.shape))


def make_simple_cost(horizon: int, batch_size: int, n_tau: int) -> tuple[torch.Tensor, torch.Tensor]:
    q_diag = torch.ones(horizon, batch_size, n_tau) * 0.1
    q_diag[:, :, 10:] = 0.01
    q_mat = torch.diag_embed(q_diag)
    p_vec = torch.zeros(horizon, batch_size, n_tau)
    return q_mat, p_vec


def validate_mpc_solve() -> None:
    import drone
    import il_env

    horizon = int(os.environ["ACMPC_T"])
    dx = drone.DroneDx(device="cpu")
    env = il_env.IL_Env("drone", mpc_T=horizon, lqr_iter=5)

    for batch_size in (1, 8):
        xinit = torch.zeros(batch_size, dx.n_state)
        xinit[:, 3] = 1.0
        q_mat, p_vec = make_simple_cost(horizon, batch_size, dx.n_state + dx.n_ctrl)

        x_mpc, u_mpc = env.mpc(dx, xinit, q_mat, p_vec, lqr_iter_override=1)
        assert_shape("x_mpc", x_mpc, (horizon, batch_size, dx.n_state))
        assert_shape("u_mpc", u_mpc, (horizon, batch_size, dx.n_ctrl))
        assert_finite("x_mpc", x_mpc)
        assert_finite("u_mpc", u_mpc)

        lower = torch.tensor([dx.thrust_min * 4, -dx.omega_max[0], -dx.omega_max[1], -dx.omega_max[2]])
        upper = torch.tensor([dx.thrust_max * 4, dx.omega_max[0], dx.omega_max[1], dx.omega_max[2]])
        if not torch.all(u_mpc >= lower.view(1, 1, -1) - 1e-4):
            raise RuntimeError(f"MPC action below lower bound for batch {batch_size}:\n{u_mpc}")
        if not torch.all(u_mpc <= upper.view(1, 1, -1) + 1e-4):
            raise RuntimeError(f"MPC action above upper bound for batch {batch_size}:\n{u_mpc}")

    print("IL_Env.mpc finite and bounded actions: ok")


def build_policy(observation_dim: int = 36):
    from training_modules.mlp_mpc_policy import MlpMpcPolicy

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

    with contextlib.redirect_stdout(io.StringIO()):
        policy = MlpMpcPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: 1e-3,
        )
    return policy


def validate_policy_forward_actor() -> None:
    horizon = int(os.environ["ACMPC_T"])

    for batch_size in (1, 8, 64):
        policy = build_policy()
        feature_device = next(policy.mlp_extractor.policy_net.parameters()).device
        dx = policy.mlp_extractor.dx

        features = torch.zeros(batch_size, policy.features_dim, device=feature_device)
        states = torch.zeros(batch_size, dx.n_state, device=feature_device)
        states[:, 3] = 1.0

        actions = policy.mlp_extractor.forward_actor(features, states)
        assert_shape("MlpMpcPolicy.forward_actor output", actions, (batch_size, 4))
        assert_finite("MlpMpcPolicy.forward_actor output", actions)

        if not torch.all(actions >= -1.0 - 1e-4):
            raise RuntimeError(f"normalized policy action below -1 for batch {batch_size}:\n{actions}")
        if not torch.all(actions <= 1.0 + 1e-4):
            raise RuntimeError(f"normalized policy action above 1 for batch {batch_size}:\n{actions}")

        predictions = policy.mlp_extractor.predictions
        expected_prediction_shape = (batch_size, horizon, dx.n_state + dx.n_ctrl)
        assert_shape("MlpMpcPolicy predictions", predictions, expected_prediction_shape)
        assert_finite("MlpMpcPolicy predictions", predictions)

        print(
            "MlpMpcPolicy.forward_actor: ok",
            "batch=",
            batch_size,
            "actions=",
            tuple(actions.shape),
            "predictions=",
            tuple(predictions.shape),
        )


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("torch cuda available:", torch.cuda.is_available())
    print("ACMPC_T:", os.environ["ACMPC_T"])

    validate_drone_forward_and_quaternion_norm()
    validate_analytic_jacobians()
    validate_mpc_solve()
    validate_policy_forward_actor()

    print("AC-MPC core validation: passed")


if __name__ == "__main__":
    main()
