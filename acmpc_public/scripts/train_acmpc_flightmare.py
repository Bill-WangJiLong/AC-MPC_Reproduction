"""Train AC-MPC with PPO on the modified Flightmare racing environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch as th


REPO_ROOT = Path(__file__).resolve().parents[1]
DIFF_MPC_DRONES = REPO_ROOT / "diff_mpc_drones"

for path in (REPO_ROOT, DIFF_MPC_DRONES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_acmpc_gym import (  # noqa: E402
    CudaMemoryCsvCallback,
    EpisodeCsvCallback,
    linear_schedule,
    positive_int,
    to_jsonable,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AC-MPC PPO on modified Flightmare RacingEnv_v1.")
    parser.add_argument("--flightmare-path", type=Path, default=Path(r"D:\MyProjects\flightmare"))
    parser.add_argument(
        "--racing-config-path",
        type=Path,
        default=None,
        help=(
            "Runtime racing_env.yaml. The current RacingEnv_v1 binary only supports "
            "<flightmare-path>/flightlib/configs/racing_env.yaml; use the track install "
            "script to switch tracks in that file."
        ),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "acmpc_flightmare")

    parser.add_argument("--acmpc-t", type=positive_int, default=2)
    parser.add_argument("--total-timesteps", type=positive_int, default=200_000)
    parser.add_argument("--n-envs", type=positive_int, default=8)
    parser.add_argument("--num-threads", type=positive_int, default=8)
    parser.add_argument("--n-steps", type=positive_int, default=250)
    parser.add_argument("--batch-size", type=positive_int, default=None)
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

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene-id", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="Pass render: yes to vec_env.yaml. Unity is not required.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-normalize-obs", action="store_true")
    parser.add_argument("--normalize-reward", action="store_true")
    parser.add_argument("--clip-obs", type=float, default=10.0)

    parser.add_argument("--checkpoint-freq", type=positive_int, default=25_000)
    parser.add_argument("--resume-model", type=Path, default=None)
    parser.add_argument("--resume-vecnormalize", type=Path, default=None)
    parser.add_argument("--cuda-memory-log", action="store_true")
    parser.add_argument("--cuda-memory-log-freq-rollouts", type=positive_int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--log-interval", type=positive_int, default=1)
    return parser


def make_env(args: argparse.Namespace, run_dir: Path):
    from stable_baselines3.common.vec_env import VecNormalize

    from acmpc_flightmare.track import load_flightmare_track, load_racing_metadata
    from acmpc_flightmare.vec_env import FlightmareRacingVecEnv, write_vec_env_config

    racing_config_path = args.racing_config_path
    if racing_config_path is None:
        racing_config_path = args.flightmare_path / "flightlib" / "configs" / "racing_env.yaml"
    racing_config_path = racing_config_path.resolve()
    vec_config_path = write_vec_env_config(
        run_dir / "flightmare_vec_env.yaml",
        seed=args.seed,
        num_envs=args.n_envs,
        num_threads=args.num_threads,
        render=args.render,
        scene_id=args.scene_id,
    )

    raw_env = FlightmareRacingVecEnv(
        flightmare_path=args.flightmare_path,
        vec_config_path=vec_config_path,
        racing_config_path=racing_config_path,
    )
    if args.resume_vecnormalize is not None:
        env = VecNormalize.load(str(args.resume_vecnormalize), raw_env)
        env.training = True
        env.norm_reward = args.normalize_reward
        env.state_space = raw_env.state_space
    elif args.no_normalize_obs and not args.normalize_reward:
        env = raw_env
    else:
        env = VecNormalize(
            raw_env,
            norm_obs=not args.no_normalize_obs,
            norm_reward=args.normalize_reward,
            clip_obs=args.clip_obs,
            training=True,
        )
        env.state_space = raw_env.state_space

    env_info = {
        "flightmare_path": str(args.flightmare_path.resolve()),
        "racing_config_path": str(racing_config_path),
        "vec_config_path": str(vec_config_path.resolve()),
        "track": load_flightmare_track(racing_config_path),
        "metadata": load_racing_metadata(racing_config_path),
        "num_envs": raw_env.num_envs,
        "obs_dim": raw_env.obs_dim,
        "act_dim": raw_env.act_dim,
        "state_dim": raw_env.state_dim,
        "extra_info_names": raw_env.extra_info_names,
        "normalize_obs": not args.no_normalize_obs,
        "normalize_reward": args.normalize_reward,
    }
    return env, env_info


def save_run_config(run_dir: Path, args: argparse.Namespace, env_info: Dict[str, Any], ppo_config: Dict[str, Any]) -> None:
    config = {
        "args": to_jsonable(vars(args)),
        "env_config": to_jsonable(env_info),
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
    args = build_arg_parser().parse_args()
    args.flightmare_path = args.flightmare_path.resolve()
    if args.racing_config_path is not None:
        args.racing_config_path = args.racing_config_path.resolve()

    os.environ["ACMPC_T"] = str(args.acmpc_t)
    os.environ["FLIGHTMARE_PATH"] = str(args.flightmare_path)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{timestamp}_flightmare_T{args.acmpc_t}"
    run_dir = args.output_dir / run_name
    checkpoint_dir = run_dir / "checkpoints"
    sb3_log_dir = run_dir / "sb3"
    csv_dir = run_dir / "csv"
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    env, env_info = make_env(args, run_dir)
    actual_n_envs = int(env.num_envs)
    rollout_size = actual_n_envs * args.n_steps
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
    save_run_config(run_dir, args, env_info, ppo_config)

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

    checkpoint_save_freq = max(args.checkpoint_freq // actual_n_envs, 1)
    callback_items = [
        CheckpointCallback(
            save_freq=checkpoint_save_freq,
            save_path=str(checkpoint_dir),
            name_prefix="acmpc_flightmare",
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
    print(f"Flightmare path: {args.flightmare_path}")
    print(f"Racing config: {env_info['racing_config_path']}")
    print(f"Track: {env_info['metadata']['track_name']}")
    print(f"ACMPC_T: {os.environ['ACMPC_T']}")
    print(f"n_envs={actual_n_envs}, n_steps={args.n_steps}, rollout_size={rollout_size}, batch_size={batch_size}")
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
