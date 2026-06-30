"""Train AC-MPC with PPO on the Python racing Gym environment.

The script is intentionally an entrypoint only. Importing it must not start
training; call it as a program to run PPO.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch as th


REPO_ROOT = Path(__file__).resolve().parents[1]
DIFF_MPC_DRONES = REPO_ROOT / "diff_mpc_drones"
TRACK_ASSET_DIR = REPO_ROOT / "acmpc_racing_gym" / "tracks" / "assets"

for path in (REPO_ROOT, DIFF_MPC_DRONES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class EpisodeCsvCallback:
    """Small SB3 callback that logs per-episode outcomes to CSV."""

    def __init__(self, csv_path: Path, verbose: int = 0):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: "EpisodeCsvCallback"):
                super().__init__(verbose=outer.verbose)
                self.outer = outer

            def _on_training_start(self) -> None:
                self.outer.on_training_start(self.training_env.num_envs)

            def _on_step(self) -> bool:
                return self.outer.on_step(self)

            def _on_training_end(self) -> None:
                self.outer.close()

        self.csv_path = csv_path
        self.verbose = verbose
        self.callback = _Callback(self)
        self._file = None
        self._writer = None
        self._episode_returns: Optional[np.ndarray] = None
        self._episode_lengths: Optional[np.ndarray] = None
        self._episode_counts: Optional[np.ndarray] = None

    def as_sb3_callback(self):
        return self.callback

    def on_training_start(self, n_envs: int) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        fieldnames = [
            "global_timesteps",
            "env_index",
            "episode_index",
            "episode_return",
            "episode_length",
            "gate_index",
            "finished",
            "collision",
            "timeout",
            "out_of_bounds",
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._episode_returns = np.zeros(n_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(n_envs, dtype=np.int64)
        self._episode_counts = np.zeros(n_envs, dtype=np.int64)

    def on_step(self, callback) -> bool:
        rewards = np.asarray(callback.locals.get("rewards", []), dtype=np.float64)
        dones = np.asarray(callback.locals.get("dones", []), dtype=bool)
        infos = callback.locals.get("infos", [])
        if self._writer is None or self._episode_returns is None or self._episode_lengths is None:
            return True
        if rewards.size == 0:
            return True

        self._episode_returns[: rewards.shape[0]] += rewards
        self._episode_lengths[: rewards.shape[0]] += 1

        for env_idx, done in enumerate(dones):
            if not done:
                continue
            info = infos[env_idx] if env_idx < len(infos) else {}
            self._episode_counts[env_idx] += 1
            self._writer.writerow(
                {
                    "global_timesteps": int(callback.num_timesteps),
                    "env_index": int(env_idx),
                    "episode_index": int(self._episode_counts[env_idx]),
                    "episode_return": float(self._episode_returns[env_idx]),
                    "episode_length": int(self._episode_lengths[env_idx]),
                    "gate_index": int(info.get("gate_index", -1)),
                    "finished": bool(info.get("finished", False)),
                    "collision": bool(info.get("collision", False)),
                    "timeout": bool(info.get("timeout", False)),
                    "out_of_bounds": bool(info.get("out_of_bounds", False)),
                }
            )
            self._file.flush()
            self._episode_returns[env_idx] = 0.0
            self._episode_lengths[env_idx] = 0
        return True

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class CudaMemoryCsvCallback:
    """Log CUDA allocator stats at rollout boundaries for leak diagnosis."""

    def __init__(self, csv_path: Path, log_freq_rollouts: int = 1, reset_peak: bool = True, verbose: int = 0):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer: "CudaMemoryCsvCallback"):
                super().__init__(verbose=outer.verbose)
                self.outer = outer

            def _on_training_start(self) -> None:
                self.outer.on_training_start()

            def _on_step(self) -> bool:
                return True

            def _on_rollout_end(self) -> None:
                self.outer.on_rollout_end(self)

            def _on_training_end(self) -> None:
                self.outer.close()

        self.csv_path = csv_path
        self.log_freq_rollouts = max(int(log_freq_rollouts), 1)
        self.reset_peak = reset_peak
        self.verbose = verbose
        self.callback = _Callback(self)
        self._file = None
        self._writer = None
        self._rollout_count = 0
        self._enabled = False

    def as_sb3_callback(self):
        return self.callback

    def on_training_start(self) -> None:
        self._enabled = th.cuda.is_available()
        if not self._enabled:
            if self.verbose:
                print("CUDA memory logging requested, but CUDA is not available.")
            return

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        fieldnames = [
            "global_timesteps",
            "rollout_index",
            "allocated_mb",
            "reserved_mb",
            "max_allocated_mb",
            "max_reserved_mb",
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        th.cuda.reset_peak_memory_stats()

    def on_rollout_end(self, callback) -> None:
        if not self._enabled or self._writer is None:
            return
        self._rollout_count += 1
        if self._rollout_count % self.log_freq_rollouts != 0:
            return

        th.cuda.synchronize()
        bytes_to_mb = 1024.0 * 1024.0
        self._writer.writerow(
            {
                "global_timesteps": int(callback.num_timesteps),
                "rollout_index": int(self._rollout_count),
                "allocated_mb": th.cuda.memory_allocated() / bytes_to_mb,
                "reserved_mb": th.cuda.memory_reserved() / bytes_to_mb,
                "max_allocated_mb": th.cuda.max_memory_allocated() / bytes_to_mb,
                "max_reserved_mb": th.cuda.max_memory_reserved() / bytes_to_mb,
            }
        )
        self._file.flush()
        if self.reset_peak:
            th.cuda.reset_peak_memory_stats()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def linear_schedule(start: float, end: float):
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining

    return schedule


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AC-MPC PPO on the Python Gym racing environment.")
    available_tracks = sorted(path.stem for path in TRACK_ASSET_DIR.glob("*.json"))

    parser.add_argument(
        "--track-name",
        default="horizontal",
        choices=available_tracks,
    )
    parser.add_argument("--single-gate", action="store_true", help="Use only the first gate from the selected track.")
    parser.add_argument("--run-name", default=None, help="Run folder name. Defaults to timestamp plus track name.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "acmpc_gym")

    parser.add_argument("--acmpc-t", type=positive_int, default=2, help="MPC horizon passed through ACMPC_T.")
    parser.add_argument("--total-timesteps", type=positive_int, default=200_000)
    parser.add_argument("--n-envs", type=positive_int, default=8)
    parser.add_argument("--n-steps", type=positive_int, default=250, help="PPO rollout steps per environment.")
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=None,
        help="PPO minibatch size. Defaults to min(25000, n_envs * n_steps).",
    )
    parser.add_argument("--n-epochs", type=positive_int, default=10)

    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--learning-rate-start", type=float, default=3e-4)
    parser.add_argument("--learning-rate-end", type=float, default=1e-5)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-std-init", type=float, default=-1.2)

    parser.add_argument("--max-episode-steps", type=positive_int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--no-normalize-obs", action="store_true", help="Disable VecNormalize observation scaling.")
    parser.add_argument("--normalize-reward", action="store_true", help="Enable VecNormalize reward scaling.")
    parser.add_argument("--clip-obs", type=float, default=10.0)

    parser.add_argument("--checkpoint-freq", type=positive_int, default=25_000)
    parser.add_argument("--resume-model", type=Path, default=None)
    parser.add_argument("--resume-vecnormalize", type=Path, default=None)
    parser.add_argument("--cuda-memory-log", action="store_true", help="Write CUDA allocator stats to csv/cuda_memory.csv.")
    parser.add_argument(
        "--cuda-memory-log-freq-rollouts",
        type=positive_int,
        default=1,
        help="Rollout interval for CUDA memory logging.",
    )
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--log-interval", type=positive_int, default=1)
    return parser


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def make_single_gate_track(track_name: str, run_dir: Path) -> Path:
    from acmpc_racing_gym.tracks.loader import ASSET_DIR

    source_path = ASSET_DIR / f"{track_name}.json"
    data = json.loads(source_path.read_text(encoding="utf-8"))
    data["name"] = f"{track_name}_single_gate"
    data["gates"] = data["gates"][:1]
    first_gate = data["gates"][0]
    gate_center = np.asarray(first_gate["center"], dtype=np.float64)
    gate_normal = np.asarray(first_gate["normal"], dtype=np.float64)
    gate_normal /= np.linalg.norm(gate_normal)
    data["finish"] = {
        "position": (gate_center + 2.0 * gate_normal).tolist(),
        "radius": float(data.get("finish", {}).get("radius", 0.5)),
    }
    output_path = run_dir / f"{track_name}_single_gate.json"
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def make_env(args: argparse.Namespace, run_dir: Path):
    from stable_baselines3.common.vec_env import VecNormalize

    from acmpc_racing_gym.config import RacingEnvConfig
    from acmpc_racing_gym.wrappers import make_acmpc_racing_vec_env

    config = RacingEnvConfig()
    config.track_name = args.track_name
    config.max_episode_steps = args.max_episode_steps
    config.seed = args.seed
    config.observation.normalize = not args.no_normalize_obs

    if args.single_gate:
        config.track_path = make_single_gate_track(args.track_name, run_dir)
        config.track_name = f"{args.track_name}_single_gate"

    if args.resume_vecnormalize is not None:
        raw_env = make_acmpc_racing_vec_env(
            n_envs=args.n_envs,
            config=config,
            normalize_obs=False,
            normalize_reward=False,
            training=True,
        )
        env = VecNormalize.load(str(args.resume_vecnormalize), raw_env)
        env.training = True
        env.norm_reward = args.normalize_reward
        env.state_space = raw_env.state_space
    else:
        env = make_acmpc_racing_vec_env(
            n_envs=args.n_envs,
            config=config,
            normalize_obs=not args.no_normalize_obs,
            normalize_reward=args.normalize_reward,
            clip_obs=args.clip_obs,
            training=True,
        )
    return env, config


def save_run_config(run_dir: Path, args: argparse.Namespace, env_config, ppo_config: Dict[str, Any]) -> None:
    config = {
        "args": to_jsonable(vars(args)),
        "env_config": to_jsonable(env_config),
        "ppo_config": to_jsonable(ppo_config),
        "versions": {
            "python": sys.version,
            "torch": th.__version__,
            "cuda_available": th.cuda.is_available(),
            "cuda_version": th.version.cuda,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    os.environ["ACMPC_T"] = str(args.acmpc_t)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{timestamp}_{args.track_name}_T{args.acmpc_t}"
    if args.single_gate:
        run_name += "_single_gate"
    run_dir = args.output_dir / run_name
    checkpoint_dir = run_dir / "checkpoints"
    sb3_log_dir = run_dir / "sb3"
    csv_dir = run_dir / "csv"
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    env, env_config = make_env(args, run_dir)
    rollout_size = args.n_envs * args.n_steps
    batch_size = args.batch_size or min(25_000, rollout_size)
    if rollout_size % batch_size != 0:
        print(
            f"Warning: rollout_size={rollout_size} is not divisible by batch_size={batch_size}; "
            "SB3 will use a truncated final minibatch.",
            file=sys.stderr,
        )

    learning_rate = linear_schedule(args.learning_rate_start, args.learning_rate_end)
    ppo_config = {
        "learning_rate_start": args.learning_rate_start,
        "learning_rate_end": args.learning_rate_end,
        "n_steps": args.n_steps,
        "batch_size": batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "log_std_init": args.log_std_init,
    }
    save_run_config(run_dir, args, env_config, ppo_config)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.logger import configure

    from training_modules.mlp_mpc_policy import MlpMpcPolicy

    if args.resume_model is not None:
        model = PPO.load(str(args.resume_model), env=env, device=args.device)
    else:
        model = PPO(
            MlpMpcPolicy,
            env,
            learning_rate=learning_rate,
            n_steps=args.n_steps,
            batch_size=batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            tensorboard_log=str(sb3_log_dir),
            policy_kwargs={"log_std_init": args.log_std_init},
            verbose=args.verbose,
            seed=args.seed,
            device=args.device,
        )

    model.set_logger(configure(str(sb3_log_dir), ["stdout", "csv", "tensorboard"]))

    checkpoint_save_freq = max(args.checkpoint_freq // args.n_envs, 1)
    callback_items = [
        CheckpointCallback(
            save_freq=checkpoint_save_freq,
            save_path=str(checkpoint_dir),
            name_prefix="acmpc_gym",
            save_vecnormalize=True,
            verbose=args.verbose,
        ),
        EpisodeCsvCallback(csv_dir / "episodes.csv", verbose=args.verbose).as_sb3_callback(),
    ]
    if args.cuda_memory_log:
        callback_items.append(
            CudaMemoryCsvCallback(
                csv_dir / "cuda_memory.csv",
                log_freq_rollouts=args.cuda_memory_log_freq_rollouts,
                verbose=args.verbose,
            ).as_sb3_callback()
        )
    callbacks = CallbackList(callback_items)

    print(f"Run directory: {run_dir}")
    print(f"Track: {env_config.track_name}")
    print(f"ACMPC_T: {os.environ['ACMPC_T']}")
    print(f"n_envs={args.n_envs}, n_steps={args.n_steps}, rollout_size={rollout_size}, batch_size={batch_size}")
    print("Starting PPO training...")
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks, log_interval=args.log_interval)
    finally:
        model.save(str(run_dir / "final_model.zip"))
        vec_norm = model.get_vec_normalize_env()
        if vec_norm is not None:
            vec_norm.save(str(run_dir / "vecnormalize.pkl"))
        env.close()
        print(f"Saved final model to: {run_dir / 'final_model.zip'}")
        if (run_dir / "vecnormalize.pkl").exists():
            print(f"Saved VecNormalize stats to: {run_dir / 'vecnormalize.pkl'}")


if __name__ == "__main__":
    main()
