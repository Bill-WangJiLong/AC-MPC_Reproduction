"""Smoke tests for the AC-MPC Python racing Gym environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DIFF_MPC_DRONES = REPO_ROOT / "diff_mpc_drones"

for path in (REPO_ROOT, DIFF_MPC_DRONES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("ACMPC_T", "2")


def assert_finite(name: str, value) -> None:
    arr = np.asarray(value)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"{name} contains non-finite values: {arr}")


def test_gate_geometry() -> None:
    from acmpc_racing_gym.tracks.gate import Gate

    gate = Gate(
        center=np.array([0.0, 0.0, 1.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        up=np.array([0.0, 0.0, 1.0]),
        width=1.0,
        height=1.0,
        frame_thickness=0.1,
    )

    assert gate.check_pass(np.array([-1.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0]))
    assert not gate.check_pass(np.array([1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0]))
    assert gate.check_frame_collision(
        np.array([-1.0, 0.55, 1.0]),
        np.array([1.0, 0.55, 1.0]),
        drone_radius=0.0,
    )
    print("gate geometry: ok")


def test_env_spaces_and_rollout() -> None:
    from acmpc_racing_gym import AcMpcRacingEnv
    from acmpc_racing_gym.config import RacingEnvConfig
    from acmpc_racing_gym.observations import TrackObservationMode

    for mode in (
        TrackObservationMode.VEHICLE_RELATIVE,
        TrackObservationMode.CHAINED_GATE_RELATIVE,
    ):
        config = RacingEnvConfig(random_reset=False, max_episode_steps=60)
        config.observation.track_obs_mode = mode
        env = AcMpcRacingEnv(config)
        obs = env.reset()
        state = env.get_state()
        assert obs.shape == (36,)
        assert state.shape == (13,)
        assert env.observation_space.shape == (36,)
        assert env.action_space.shape == (4,)
        assert_finite("obs", obs)
        assert_finite("state", state)

    env = AcMpcRacingEnv(RacingEnvConfig(random_reset=True, max_episode_steps=80))
    obs = env.reset()
    ret = 0.0
    done = False
    info = {}
    for step in range(80):
        obs, reward, done, info = env.step(env.action_space.sample())
        state = env.get_state()
        assert obs.shape == (36,)
        assert state.shape == (13,)
        assert_finite("rollout obs", obs)
        assert_finite("rollout state", state)
        assert np.isfinite(reward)
        ret += reward
        if done:
            break
    if not done:
        raise RuntimeError("Random rollout did not terminate within max_episode_steps")
    print("env spaces and random rollout: ok", "steps=", step + 1, "return=", round(ret, 4), "info=", info)


def test_last_gate_and_finish_in_same_step() -> None:
    from acmpc_racing_gym import AcMpcRacingEnv
    from acmpc_racing_gym.config import RacingEnvConfig

    config = RacingEnvConfig(track_name="vertical", random_reset=False)
    env = AcMpcRacingEnv(config)
    env.track.current_index = len(env.track.gates) - 1
    env.dynamics.state.position = np.array([0.0, 0.0, 1.2], dtype=np.float64)

    prediction = np.zeros((1, 14), dtype=np.float64)
    prediction[0, 0:3] = [0.0, 0.0, 0.4]
    prediction[0, 3] = 1.0
    prediction_result = env.compute_prediction_rollout(prediction)
    assert prediction_result["valid"][0] == 1.0
    assert prediction_result["terminal"][0] == 1.0
    assert np.isclose(
        prediction_result["rewards"][0],
        config.reward.gate_pass_reward + config.reward.finish_reward,
    )
    assert env.track.current_index == len(env.track.gates) - 1

    def cross_last_gate_and_finish(_action):
        state = env.dynamics.state.copy()
        state.position = np.array([0.0, 0.0, 0.4], dtype=np.float64)
        env.dynamics.state = state
        return state.copy(), env.dynamics.last_command

    env.dynamics.step = cross_last_gate_and_finish
    _, reward, done, info = env.step(np.zeros(4, dtype=np.float32))
    assert info["gate_passed"]
    assert info["finished"]
    assert not info["collision"]
    assert done
    assert np.isclose(
        reward,
        config.reward.gate_pass_reward + config.reward.finish_reward,
    )
    env.close()
    print("same-step last gate and finish: ok")


def test_vec_env_state() -> None:
    from acmpc_racing_gym.config import RacingEnvConfig
    from acmpc_racing_gym.wrappers import make_acmpc_racing_vec_env

    env = make_acmpc_racing_vec_env(n_envs=2, config=RacingEnvConfig(max_episode_steps=20))
    obs = env.reset()
    states = env.get_state()
    assert obs.shape == (2, 36)
    assert states.shape == (2, 13)
    assert env.state_space.shape == (13,)
    assert_finite("vec obs", obs)
    assert_finite("vec states", states)
    obs, reward, done, info = env.step(np.zeros((2, 4), dtype=np.float32))
    assert obs.shape == (2, 36)
    assert states.shape == (2, 13)
    assert reward.shape == (2,)
    assert done.shape == (2,)
    env.close()
    print("state vec env: ok")


def test_vec_env_observation_normalization() -> None:
    from stable_baselines3.common.vec_env import VecNormalize

    from acmpc_racing_gym.config import RacingEnvConfig
    from acmpc_racing_gym.wrappers import make_acmpc_racing_vec_env

    config = RacingEnvConfig(max_episode_steps=20)
    env = make_acmpc_racing_vec_env(n_envs=4, config=config)
    if not isinstance(env, VecNormalize):
        raise RuntimeError(f"Expected VecNormalize by default, got {type(env)}")

    obs = env.reset()
    states = env.get_state()
    assert obs.shape == (4, 36)
    assert states.shape == (4, 13)
    assert env.state_space.shape == (13,)
    assert_finite("normalized vec obs", obs)
    assert_finite("normalized vec states", states)
    original_obs = env.get_original_obs()
    assert original_obs.shape == (4, 36)
    assert_finite("original vec obs", original_obs)
    if np.allclose(obs, original_obs):
        raise RuntimeError("Normalized observation unexpectedly equals original observation")
    env.close()

    raw_env = make_acmpc_racing_vec_env(n_envs=2, config=config, normalize_obs=False)
    if isinstance(raw_env, VecNormalize):
        raise RuntimeError("Expected raw vec env when normalize_obs=False")
    raw_obs = raw_env.reset()
    raw_states = raw_env.get_state()
    assert raw_obs.shape == (2, 36)
    assert raw_states.shape == (2, 13)
    assert_finite("raw vec obs", raw_obs)
    raw_env.close()
    print("vec observation normalization: ok")


def test_policy_forward_from_env() -> None:
    from acmpc_racing_gym import AcMpcRacingEnv
    from acmpc_racing_gym.config import RacingEnvConfig
    from training_modules.mlp_mpc_policy import MlpMpcPolicy

    env = AcMpcRacingEnv(RacingEnvConfig(random_reset=False))
    obs = env.reset()
    state = env.get_state()

    policy = MlpMpcPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 1e-3,
    )
    feature_device = next(policy.mlp_extractor.policy_net.parameters()).device
    obs_tensor = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=feature_device)
    state_tensor = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32, device=feature_device)
    with torch.no_grad():
        action = policy.mlp_extractor.forward_actor(obs_tensor, state_tensor)
    if tuple(action.shape) != (1, 4):
        raise RuntimeError(f"Unexpected AC-MPC action shape: {tuple(action.shape)}")
    if not torch.isfinite(action).all():
        raise RuntimeError(f"AC-MPC action contains non-finite values: {action}")
    if not torch.all(action >= -1.0 - 1e-4) or not torch.all(action <= 1.0 + 1e-4):
        raise RuntimeError(f"AC-MPC normalized action out of bounds: {action}")
    print("policy forward from env: ok", tuple(action.shape))


def test_sb3_state_plumbing() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import obs_as_tensor

    from acmpc_racing_gym.config import RacingEnvConfig
    from acmpc_racing_gym.wrappers import make_acmpc_racing_vec_env
    from training_modules.mlp_mpc_policy import MlpMpcPolicy

    env = make_acmpc_racing_vec_env(n_envs=1, config=RacingEnvConfig(max_episode_steps=20))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO(
        MlpMpcPolicy,
        env,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        learning_rate=1e-4,
        verbose=0,
        device=device,
    )

    obs = env.reset()
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    model.rollout_buffer.reset()

    for _ in range(2):
        state_np = env.get_state()
        obs_tensor = obs_as_tensor(obs, model.device)
        state_tensor = obs_as_tensor(state_np, model.device)
        with torch.no_grad():
            actions, values, log_probs = model.policy(obs_tensor, state_tensor)
        actions_np = actions.cpu().numpy()
        new_obs, rewards, dones, infos = env.step(np.clip(actions_np, env.action_space.low, env.action_space.high))
        model.rollout_buffer.add(obs, actions_np, state_np, rewards, episode_starts, values, log_probs)
        obs = new_obs
        episode_starts = dones

    if model.rollout_buffer.states.shape != (2, 1, 13):
        raise RuntimeError(f"Unexpected rollout state shape: {model.rollout_buffer.states.shape}")

    batch = next(model.rollout_buffer.get(batch_size=2))
    with torch.no_grad():
        values, log_prob, entropy = model.policy.evaluate_actions(batch.observations, batch.actions, batch.states)
    if tuple(values.shape) != (2, 1):
        raise RuntimeError(f"Unexpected value shape from evaluate_actions: {tuple(values.shape)}")
    if not torch.isfinite(log_prob).all():
        raise RuntimeError("Non-finite log_prob from evaluate_actions")
    del entropy
    env.close()
    print("SB3 state plumbing: ok", "states=", (2, 1, 13))


def main() -> None:
    np.random.seed(0)
    torch.manual_seed(0)
    print("ACMPC_T:", os.environ["ACMPC_T"])
    test_gate_geometry()
    test_env_spaces_and_rollout()
    test_last_gate_and_finish_in_same_step()
    test_vec_env_state()
    test_vec_env_observation_normalization()
    test_policy_forward_from_env()
    test_sb3_state_plumbing()
    print("AC-MPC racing Gym smoke test: passed")


if __name__ == "__main__":
    main()
