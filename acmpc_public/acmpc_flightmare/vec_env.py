"""SB3 VecEnv adapter for the modified Flightmare RacingEnv_v1."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from .track import load_flightmare_track, load_racing_metadata


VecEnvIndices = Union[None, int, Sequence[int]]

BOOL_INFO_KEYS = {"finished", "collision", "gate_passed", "finish_phase", "out_of_bounds", "timeout"}
INT_INFO_KEYS = {"gate_index", "collision_code"}
COLLISION_CODES = {
    0: "",
    1: "out_of_bounds",
    2: "ground",
    3: "gate_frame",
    4: "invalid_state",
}


def default_flightmare_path() -> Path:
    value = os.environ.get("FLIGHTMARE_PATH")
    if value:
        return Path(value).resolve()
    return Path(r"D:\MyProjects\flightmare").resolve()


def ensure_flightgym_importable(flightmare_path: Path) -> None:
    flightmare_path = flightmare_path.resolve()
    flightlib_path = flightmare_path / "flightlib"
    os.environ["FLIGHTMARE_PATH"] = str(flightmare_path)
    for path in (flightmare_path, flightlib_path):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def write_vec_env_config(
    path: Path,
    *,
    seed: int,
    num_envs: int,
    num_threads: int,
    render: bool = False,
    scene_id: int = 0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    render_value = "yes" if render else "no"
    text = (
        "env:\n"
        f"  seed: {int(seed)}\n"
        f"  scene_id: {int(scene_id)}\n"
        f"  num_envs: {int(num_envs)}\n"
        f"  num_threads: {int(num_threads)}\n"
        f"  render: {render_value}\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


class FlightmareRacingVecEnv(VecEnv):
    """Wrap Flightmare's already-vectorized RacingEnv_v1 as an SB3 VecEnv."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        flightmare_path: Optional[Path] = None,
        vec_config_path: Optional[Path] = None,
        racing_config_path: Optional[Path] = None,
    ):
        self.flightmare_path = (flightmare_path or default_flightmare_path()).resolve()
        ensure_flightgym_importable(self.flightmare_path)

        import flightgym  # type: ignore

        self.vec_config_path = vec_config_path.resolve() if vec_config_path is not None else None
        runtime_racing_config = (
            self.flightmare_path / "flightlib" / "configs" / "racing_env.yaml"
        ).resolve()
        requested_racing_config = (
            racing_config_path.resolve()
            if racing_config_path is not None
            else runtime_racing_config
        )
        if requested_racing_config != runtime_racing_config:
            raise ValueError(
                "The current RacingEnv_v1 binary constructs each RacingEnv with its "
                "default constructor and therefore only reads "
                f"{runtime_racing_config}. Install/switch the track in that runtime "
                "config before training; passing a different racing_config_path would "
                "only change Python metadata."
            )
        self.racing_config_path = runtime_racing_config
        self.metadata_dict = load_racing_metadata(self.racing_config_path)
        self.track = load_flightmare_track(self.racing_config_path)

        if self.vec_config_path is None:
            self._env = flightgym.RacingEnv_v1()
        else:
            self._env = flightgym.RacingEnv_v1(str(self.vec_config_path))

        self.n_envs = int(self._env.getNumOfEnvs())
        self.obs_dim = int(self._env.getObsDim())
        self.act_dim = int(self._env.getActDim())
        self.state_dim = int(self._env.getStateDim())
        self.extra_info_names = list(self._env.getExtraInfoNames())
        self.dt = float(self.metadata_dict["sim_dt"])
        self.max_episode_steps = int(self.metadata_dict["max_episode_steps"])

        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.act_dim,), dtype=np.float32)
        super().__init__(self.n_envs, observation_space, action_space)
        self.state_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.state_dim,), dtype=np.float32)

        self._obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        self._state = np.zeros((self.num_envs, self.state_dim), dtype=np.float32)
        self._rewards = np.zeros(self.num_envs, dtype=np.float32)
        self._dones = np.zeros(self.num_envs, dtype=np.bool_)
        self._extra = np.zeros((self.num_envs, len(self.extra_info_names)), dtype=np.float32)
        self._actions = np.zeros((self.num_envs, self.act_dim), dtype=np.float32)
        self._waiting_step = False
        self._closed = False

    def reset(self):
        self._waiting_step = False
        self._env.reset(self._obs)
        self._refresh_state()
        return self._obs.copy()

    def step_async(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.act_dim):
            actions = actions.reshape(self.num_envs, self.act_dim)
        np.clip(actions, self.action_space.low, self.action_space.high, out=self._actions)
        self._waiting_step = True

    def step_wait(self):
        if not self._waiting_step:
            raise RuntimeError("step_wait() called before step_async().")
        self._env.step(self._actions, self._obs, self._rewards, self._dones, self._extra)
        self._waiting_step = False
        self._refresh_state()
        infos = [self._make_info(index) for index in range(self.num_envs)]
        return self._obs.copy(), self._rewards.copy(), self._dones.copy(), infos

    def close(self) -> None:
        if self._closed:
            return
        close_fn = getattr(self._env, "close", None)
        if close_fn is not None:
            close_fn()
        self._closed = True

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        if seed is None:
            return [None for _ in range(self.num_envs)]
        self._env.setSeed(int(seed))
        return [int(seed) + index for index in range(self.num_envs)]

    def get_state(self) -> np.ndarray:
        self._refresh_state()
        return self._state.copy()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        selected = list(self._get_indices(indices))
        if attr_name == "track":
            return [self.track for _ in selected]
        if attr_name == "metadata_dict":
            return [self.metadata_dict for _ in selected]
        if hasattr(self, attr_name):
            value = getattr(self, attr_name)
            return [value for _ in selected]
        raise AttributeError(f"{type(self).__name__} has no attribute {attr_name!r}")

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        selected = list(self._get_indices(indices))
        if len(selected) != self.num_envs:
            raise NotImplementedError("FlightmareRacingVecEnv only supports global attributes.")
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        selected = list(self._get_indices(indices))
        if method_name == "get_state":
            state = self.get_state()
            return [state[index] for index in selected]
        if method_name == "get_track":
            return [self.track for _ in selected]
        method = getattr(self._env, method_name, None)
        if method is None:
            raise AttributeError(f"Flightmare env has no method {method_name!r}")
        result = method(*method_args, **method_kwargs)
        return [result for _ in selected]

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None) -> List[bool]:
        return [False for _ in self._get_indices(indices)]

    def get_images(self):
        raise NotImplementedError("Unity rendering is intentionally not exposed for this headless RacingEnv adapter.")

    def _refresh_state(self) -> None:
        self._env.getState(self._state)

    def _make_info(self, index: int) -> Dict[str, Any]:
        row = self._extra[index]
        info: Dict[str, Any] = {}
        for key, raw_value in zip(self.extra_info_names, row):
            value = float(raw_value)
            if key in BOOL_INFO_KEYS:
                info[key] = bool(value > 0.5)
            elif key in INT_INFO_KEYS:
                info[key] = int(round(value))
            else:
                info[key] = value

        state = self._state[index]
        # Flightmare copies extra_info before its VecEnv auto-resets a terminal
        # environment. Prefer those pre-reset values for terminal logging.
        position = np.asarray(
            [info.get("x", state[0]), info.get("y", state[1]), info.get("z", state[2])],
            dtype=float,
        )
        if not np.all(np.isfinite(position)):
            position = state[0:3].astype(float)
        info["position"] = position.tolist()
        info["quaternion"] = state[3:7].astype(float).tolist()
        info["velocity"] = state[7:10].astype(float).tolist()
        info["omega"] = state[10:13].astype(float).tolist()
        speed = float(info.get("speed", np.linalg.norm(state[7:10])))
        info["speed"] = speed if np.isfinite(speed) else float(np.linalg.norm(state[7:10]))
        collision_code = int(info.get("collision_code", 0))
        info["collision_type"] = COLLISION_CODES.get(collision_code, f"code_{collision_code}")
        if bool(info.get("out_of_bounds", False)) and not info["collision_type"]:
            info["collision_type"] = "out_of_bounds"
        if bool(self._dones[index]):
            info["terminal_observation"] = self._obs[index].copy()
        return info


def make_flightmare_racing_vec_env(
    *,
    flightmare_path: Optional[Path] = None,
    vec_config_path: Optional[Path] = None,
    racing_config_path: Optional[Path] = None,
    normalize_obs: bool = True,
    normalize_reward: bool = False,
    clip_obs: float = 10.0,
    training: bool = True,
):
    raw_env = FlightmareRacingVecEnv(
        flightmare_path=flightmare_path,
        vec_config_path=vec_config_path,
        racing_config_path=racing_config_path,
    )
    if not normalize_obs and not normalize_reward:
        return raw_env
    env = VecNormalize(
        raw_env,
        norm_obs=normalize_obs,
        norm_reward=normalize_reward,
        clip_obs=clip_obs,
        training=training,
    )
    env.state_space = raw_env.state_space
    return env
