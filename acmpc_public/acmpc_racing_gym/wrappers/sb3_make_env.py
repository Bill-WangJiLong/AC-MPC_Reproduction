"""Factory helpers for SB3 training."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Optional

from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from acmpc_racing_gym.config import RacingEnvConfig
from acmpc_racing_gym.envs.racing_env import AcMpcRacingEnv
from acmpc_racing_gym.wrappers.state_vec_env import StateDummyVecEnv


def make_acmpc_racing_env(config: Optional[RacingEnvConfig] = None) -> Callable[[], AcMpcRacingEnv]:
    def _make() -> AcMpcRacingEnv:
        return AcMpcRacingEnv(deepcopy(config) if config is not None else None)

    return _make


def make_acmpc_racing_vec_env(
    n_envs: int = 1,
    config: Optional[RacingEnvConfig] = None,
    normalize_obs: Optional[bool] = None,
    normalize_reward: bool = False,
    clip_obs: float = 10.0,
    training: bool = True,
) -> VecEnv:
    effective_config = deepcopy(config) if config is not None else RacingEnvConfig()
    env = StateDummyVecEnv([make_acmpc_racing_env(effective_config) for _ in range(n_envs)])
    use_normalize_obs = bool(effective_config.observation.normalize) if normalize_obs is None else bool(normalize_obs)
    if not use_normalize_obs:
        return env

    normalized_env = VecNormalize(
        env,
        training=training,
        norm_obs=True,
        norm_reward=normalize_reward,
        clip_obs=clip_obs,
    )
    # Make the AC-MPC state contract visible without relying on wrapper attribute recursion.
    normalized_env.state_space = env.state_space
    return normalized_env
