# AC-MPC Environment Status

Last verified in workspace:

```text
D:/MyProjects/acmpc_public
```

## Conda Environment

```text
conda env name: acmpc
python: 3.10.20
```

## Core Packages

```text
torch: 2.6.0+cu118
torch cuda runtime: 11.8
torch cuda available: True
numpy: 1.26.4
gym: 0.21.0
stable_baselines3: 1.7.0a1
mpc package path: D:/MyProjects/acmpc_public/mpc.pytorch/mpc/__init__.py
```

The installed `stable_baselines3` package resolves to the local AC-MPC fork:

```text
D:/MyProjects/acmpc_public/stable-baselines3/stable_baselines3/__init__.py
```

## Submodule Commits

```text
mpc.pytorch: 63732fa85ab2a151045493c4e67653210ca3d7ff
stable-baselines3-acmpc: 152c353863d3b05fb5feed4deb37b952bb4beb7b
```

## Environment Variables

The first-stage smoke tests use:

```text
ACMPC_T=2
PYTHONPATH includes:
  D:/MyProjects/acmpc_public
  D:/MyProjects/acmpc_public/diff_mpc_drones
```

`ACMPC_T=2` corresponds to the paper's short-horizon AC-MPC setting `N=2` and is used first because it is the cheapest configuration to validate.

## Verified Smoke Test

Command:

```powershell
conda run -n acmpc python scripts\smoke_test_acmpc_forward.py
```

Verified outputs:

```text
imports: ok
DroneDx.forward: ok (1, 10)
IL_Env.mpc: ok (2, 1, 10) (2, 1, 4)
MlpMpcPolicy init: ok ACMPC_T= 2
AC-MPC core smoke test: passed
```

Warnings seen during the smoke test:

- `torch.Tensor.T` deprecation warning from `diff_mpc_drones/drone.py`.
- `torch.lu` and `torch.lu_solve` deprecation warnings from `mpc.pytorch`.

These warnings do not block the current smoke test, but they should be tracked because PyTorch may remove those APIs in a future release.

## Verified Core Validation

Command:

```powershell
conda run -n acmpc python scripts\validate_acmpc_core.py
```

Verified checks:

```text
DroneDx.forward output is finite.
Quaternion norm after DroneDx.forward remains unit-length within 1e-4.
Analytic Jacobian shapes are A=(8, 10, 10), B=(8, 10, 4), with finite values.
IL_Env.mpc returns finite x_mpc and u_mpc.
IL_Env.mpc actions satisfy physical MPC bounds.
MlpMpcPolicy.forward_actor works for batch sizes 1, 8, and 64.
MlpMpcPolicy.forward_actor normalized actions are inside [-1, 1].
MlpMpcPolicy.mlp_extractor.predictions is finite and has shape (batch, ACMPC_T, 14).
```

Observed output:

```text
DroneDx.forward and quaternion norm: ok
analytic jacobians: ok (8, 10, 10) (8, 10, 4)
IL_Env.mpc finite and bounded actions: ok
MlpMpcPolicy.forward_actor: ok batch= 1 actions= (1, 4) predictions= (1, 2, 14)
MlpMpcPolicy.forward_actor: ok batch= 8 actions= (8, 4) predictions= (8, 2, 14)
MlpMpcPolicy.forward_actor: ok batch= 64 actions= (64, 4) predictions= (64, 2, 14)
AC-MPC core validation: passed
```

Notes:

- `MlpMpcPolicy.mlp_extractor.predictions` is initialized as a placeholder, but after `forward_actor()` it stores the solved MPC trajectory as `[x, u]`, so its runtime shape is `(batch, T, 10 + 4)`.
- `CustomNetwork.device` may be CUDA while the SB3-created `policy_net` parameters remain on CPU before an explicit `.to(device)`. The validation script creates `features` on the policy network parameter device and lets `forward_actor()` move `states` according to the repository implementation.

## Verified Python Racing Gym Smoke Test

Command:

```powershell
conda run -n acmpc python scripts\smoke_test_racing_gym.py
```

Verified checks:

```text
Gate passing and gate-frame collision geometry work.
Both track observation modes return 36-dimensional observations:
  vehicle_relative
  chained_gate_relative
Random actions can run until episode termination.
AcMpcRacingEnv observation_space is Box(shape=(36,)).
AcMpcRacingEnv action_space is Box(shape=(4,), range=[-1, 1]).
AcMpcRacingEnv.get_state() returns shape=(13,).
StateDummyVecEnv exposes get_state() with shape=(n_envs, 13).
MlpMpcPolicy.forward_actor() can consume env obs/state and return a finite action.
SB3 rollout buffer can store states with shape=(n_steps, n_envs, 13).
SB3 policy.evaluate_actions() can consume rollout observations, actions, and states.
```

Observed output:

```text
gate geometry: ok
env spaces and random rollout: ok
state vec env: ok
policy forward from env: ok (1, 4)
SB3 state plumbing: ok states= (2, 1, 13)
AC-MPC racing Gym smoke test: passed
```

Notes:

- A full `PPO.learn()` smoke was not used as a default test because AC-MPC differentiable MPC backpropagation is expensive and a 2-step run can take too long on this setup.
- The smoke test verifies the state plumbing path without running a full PPO gradient update.
