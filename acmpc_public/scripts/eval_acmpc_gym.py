"""Evaluate a trained AC-MPC PPO policy in the Python racing Gym."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic or stochastic inference for a trained AC-MPC Gym policy "
            "and save metrics plus trajectory CSV files."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Training run directory.")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the newest directory under runs/acmpc_gym when --run-dir is not set.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "runs" / "acmpc_gym",
        help="Root directory used by --latest.",
    )
    parser.add_argument("--model-path", type=Path, default=None, help="Explicit PPO model zip.")
    parser.add_argument("--vecnormalize-path", type=Path, default=None, help="Explicit VecNormalize pkl.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for evaluation outputs.")

    parser.add_argument("--episodes", type=positive_int, default=32)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--stochastic", action="store_true", help="Sample from the policy distribution.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument(
        "--model-class",
        choices=["auto", "base", "mpve"],
        default="auto",
        help="Model class used for loading. auto detects MPVE from run config.",
    )
    parser.add_argument("--acmpc-t", type=positive_int, default=None, help="Override ACMPC_T before loading policy.")
    parser.add_argument("--max-episode-steps", type=positive_int, default=None)
    parser.add_argument("--track-name", default=None)
    parser.add_argument("--track-path", type=Path, default=None)
    parser.add_argument(
        "--random-reset",
        choices=["config", "true", "false"],
        default="config",
        help="Use config random reset, force random reset, or force deterministic reset.",
    )
    parser.add_argument(
        "--allow-missing-vecnormalize",
        action="store_true",
        help="Evaluate with raw observations if VecNormalize statistics are missing.",
    )
    return parser


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def latest_run_dir(runs_root: Path) -> Path:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")
    candidates = [path for path in runs_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {runs_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_run_config(run_dir: Optional[Path]) -> Dict[str, Any]:
    if run_dir is None:
        return {}
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config does not exist: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def checkpoint_step(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.", path.name)
    return int(match.group(1)) if match else -1


def select_model_and_vecnormalize(
    run_dir: Optional[Path],
    model_path: Optional[Path],
    vecnormalize_path: Optional[Path],
) -> Tuple[Path, Optional[Path]]:
    if model_path is not None:
        selected_model = model_path
    elif run_dir is not None and (run_dir / "final_model.zip").exists():
        selected_model = run_dir / "final_model.zip"
    elif run_dir is not None:
        checkpoints = sorted(
            list((run_dir / "checkpoints").glob("acmpc_gym_*_steps.zip"))
            + list((run_dir / "checkpoints").glob("acmpc_gym_mpve_*_steps.zip"))
            + list((run_dir / "checkpoints").glob("acmpc_flightmare_*_steps.zip"))
            + list((run_dir / "checkpoints").glob("acmpc_flightmare_mpve_*_steps.zip")),
            key=checkpoint_step,
        )
        if not checkpoints:
            raise FileNotFoundError(f"No final_model.zip or checkpoint zip found in: {run_dir}")
        selected_model = checkpoints[-1]
    else:
        raise ValueError("Either --run-dir/--latest or --model-path must be provided")

    if not selected_model.exists():
        raise FileNotFoundError(f"Model file does not exist: {selected_model}")

    if vecnormalize_path is not None:
        selected_vecnormalize = vecnormalize_path
    elif run_dir is not None and (run_dir / "vecnormalize.pkl").exists():
        selected_vecnormalize = run_dir / "vecnormalize.pkl"
    elif run_dir is not None:
        step = checkpoint_step(selected_model)
        if step >= 0:
            candidates = [
                run_dir / "checkpoints" / f"acmpc_gym_vecnormalize_{step}_steps.pkl",
                run_dir / "checkpoints" / f"acmpc_gym_mpve_vecnormalize_{step}_steps.pkl",
                run_dir / "checkpoints" / f"acmpc_flightmare_vecnormalize_{step}_steps.pkl",
                run_dir / "checkpoints" / f"acmpc_flightmare_mpve_vecnormalize_{step}_steps.pkl",
            ]
            selected_vecnormalize = next((candidate for candidate in candidates if candidate.exists()), None)
        else:
            selected_vecnormalize = None
    else:
        selected_vecnormalize = None

    if selected_vecnormalize is not None and not selected_vecnormalize.exists():
        raise FileNotFoundError(f"VecNormalize file does not exist: {selected_vecnormalize}")
    return selected_model, selected_vecnormalize


def apply_config_dict(obj: Any, data: Dict[str, Any]) -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            apply_config_dict(current, value)
        elif key == "track_path":
            setattr(obj, key, Path(value) if value else None)
        elif key == "world_bounds" and value is not None:
            setattr(obj, key, tuple(tuple(float(item) for item in pair) for pair in value))
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)


def make_eval_config(args: argparse.Namespace, run_config: Dict[str, Any]):
    from acmpc_racing_gym.config import RacingEnvConfig

    config = RacingEnvConfig()
    env_config = run_config.get("env_config", {})
    if env_config:
        apply_config_dict(config, env_config)

    if args.track_name is not None:
        config.track_name = args.track_name
    if args.track_path is not None:
        config.track_path = args.track_path
    if args.max_episode_steps is not None:
        config.max_episode_steps = args.max_episode_steps
    if args.random_reset == "true":
        config.random_reset = True
    elif args.random_reset == "false":
        config.random_reset = False
    config.seed = args.seed
    return config


def make_eval_env(config: Any, vecnormalize_path: Optional[Path]):
    from stable_baselines3.common.vec_env import VecNormalize

    from acmpc_racing_gym.wrappers import make_acmpc_racing_vec_env

    raw_env = make_acmpc_racing_vec_env(
        n_envs=1,
        config=config,
        normalize_obs=False,
        normalize_reward=False,
        training=False,
    )
    if vecnormalize_path is None:
        return raw_env

    env = VecNormalize.load(str(vecnormalize_path), raw_env)
    env.training = False
    env.norm_reward = False
    env.state_space = raw_env.state_space
    return env


def serialize_track(track: Any) -> Dict[str, Any]:
    return {
        "name": track.name,
        "start": {
            "position": track.start.position.astype(float).tolist(),
            "yaw": float(track.start.yaw),
        },
        "finish": {
            "position": track.finish.position.astype(float).tolist(),
            "radius": float(track.finish.radius),
        },
        "gates": [
            {
                "index": index,
                "label": gate.label or f"G{index}",
                "center": gate.center.astype(float).tolist(),
                "normal": gate.normal.astype(float).tolist(),
                "up": gate.up.astype(float).tolist(),
                "right": gate.right.astype(float).tolist(),
                "width": float(gate.width),
                "height": float(gate.height),
                "frame_thickness": float(gate.frame_thickness),
                "corners": gate.corners_world().astype(float).tolist(),
            }
            for index, gate in enumerate(track.gates)
        ],
    }


def get_track_from_env(env: Any) -> Any:
    tracks = env.get_attr("track")
    if not tracks:
        raise RuntimeError("Could not read track from evaluation environment")
    return tracks[0]


def scalar_info(info: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = info.get(key, default)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_trajectory_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one_episode(
    env: Any,
    model: Any,
    episode_index: int,
    seed: int,
    deterministic: bool,
    max_episode_steps: int,
    dt: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    env.seed(seed)
    obs = env.reset()
    rows: List[Dict[str, Any]] = []
    total_reward = 0.0
    done = np.array([False])
    info: Dict[str, Any] = {}

    initial_state = env.get_state()[0]
    initial_speed = float(np.linalg.norm(initial_state[7:10]))
    rows.append(
        {
            "episode": episode_index,
            "step": 0,
            "time_s": 0.0,
            "reward": 0.0,
            "done": False,
            "gate_index": 0,
            "gate_passed": False,
            "collision": False,
            "collision_type": "",
            "finished": False,
            "timeout": False,
            "x": float(initial_state[0]),
            "y": float(initial_state[1]),
            "z": float(initial_state[2]),
            "vx": float(initial_state[7]),
            "vy": float(initial_state[8]),
            "vz": float(initial_state[9]),
            "speed": initial_speed,
            "wx": float(initial_state[10]),
            "wy": float(initial_state[11]),
            "wz": float(initial_state[12]),
            "action_thrust": math.nan,
            "action_wx": math.nan,
            "action_wy": math.nan,
            "action_wz": math.nan,
            "mass_normalized_thrust": math.nan,
            "collective_thrust_N": math.nan,
            "cmd_wx": math.nan,
            "cmd_wy": math.nan,
            "cmd_wz": math.nan,
        }
    )

    for step in range(1, max_episode_steps + 1):
        state_np = env.get_state()
        action, _ = model.policy.predict(obs, state_np, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, done, infos = env.step(action)
        info = infos[0]
        reward_value = float(np.asarray(reward).reshape(-1)[0])
        total_reward += reward_value

        position = np.asarray(info.get("position", [np.nan, np.nan, np.nan]), dtype=float)
        velocity = np.asarray(info.get("velocity", [np.nan, np.nan, np.nan]), dtype=float)
        omega = np.asarray(info.get("omega", [np.nan, np.nan, np.nan]), dtype=float)
        speed = float(np.linalg.norm(velocity)) if np.all(np.isfinite(velocity)) else math.nan
        command = info.get("physical_command", {}) or {}
        body_rate_cmd = np.asarray(command.get("body_rate_cmd", [np.nan, np.nan, np.nan]), dtype=float)
        action0 = action.reshape(-1)

        rows.append(
            {
                "episode": episode_index,
                "step": step,
                "time_s": step * dt,
                "reward": reward_value,
                "done": bool(done[0]),
                "gate_index": int(scalar_info(info, "gate_index", 0)),
                "gate_passed": bool(scalar_info(info, "gate_passed", False)),
                "collision": bool(scalar_info(info, "collision", False)),
                "collision_type": scalar_info(info, "collision_type", "") or "",
                "finished": bool(scalar_info(info, "finished", False)),
                "timeout": bool(scalar_info(info, "timeout", False)),
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "vx": float(velocity[0]),
                "vy": float(velocity[1]),
                "vz": float(velocity[2]),
                "speed": speed,
                "wx": float(omega[0]),
                "wy": float(omega[1]),
                "wz": float(omega[2]),
                "action_thrust": float(action0[0]),
                "action_wx": float(action0[1]),
                "action_wy": float(action0[2]),
                "action_wz": float(action0[3]),
                "mass_normalized_thrust": float(command.get("mass_normalized_thrust", math.nan)),
                "collective_thrust_N": float(command.get("collective_thrust_N", math.nan)),
                "cmd_wx": float(body_rate_cmd[0]),
                "cmd_wy": float(body_rate_cmd[1]),
                "cmd_wz": float(body_rate_cmd[2]),
            }
        )
        if bool(done[0]):
            break

    last_row = rows[-1]
    speed_values = [
        row["speed"]
        for row in rows[1:]
        if np.isfinite(row["speed"])
    ]
    success = bool(last_row["finished"])
    collision = bool(last_row["collision"])
    timeout = bool(last_row["timeout"] or (not done[0] and len(rows) > max_episode_steps))
    summary = {
        "episode": episode_index,
        "seed": seed,
        "return": total_reward,
        "length": len(rows) - 1,
        "duration_s": (len(rows) - 1) * dt,
        "success": success,
        "collision": collision,
        "timeout": timeout,
        "collision_type": str(last_row["collision_type"]),
        "final_gate_index": int(last_row["gate_index"]),
        "average_velocity": float(np.mean(speed_values)) if speed_values else math.nan,
        "lap_time": (len(rows) - 1) * dt if success else math.nan,
        "final_x": float(last_row["x"]),
        "final_y": float(last_row["y"]),
        "final_z": float(last_row["z"]),
    }
    return summary, rows


def write_summary_csv(path: Path, summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def aggregate_metrics(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    returns = np.asarray([item["return"] for item in summaries], dtype=float)
    lengths = np.asarray([item["length"] for item in summaries], dtype=float)
    velocities = np.asarray([item["average_velocity"] for item in summaries], dtype=float)
    lap_times = np.asarray([item["lap_time"] for item in summaries], dtype=float)
    successes = np.asarray([item["success"] for item in summaries], dtype=bool)
    collisions = np.asarray([item["collision"] for item in summaries], dtype=bool)
    timeouts = np.asarray([item["timeout"] for item in summaries], dtype=bool)
    gate_indices = np.asarray([item["final_gate_index"] for item in summaries], dtype=float)

    finite_lap_times = lap_times[np.isfinite(lap_times)]
    finite_velocities = velocities[np.isfinite(velocities)]
    return {
        "episodes": len(summaries),
        "return_mean": float(np.mean(returns)) if len(returns) else math.nan,
        "return_std": float(np.std(returns)) if len(returns) else math.nan,
        "success_rate": float(np.mean(successes)) if len(successes) else math.nan,
        "crash_rate": float(np.mean(collisions)) if len(collisions) else math.nan,
        "timeout_rate": float(np.mean(timeouts)) if len(timeouts) else math.nan,
        "average_length": float(np.mean(lengths)) if len(lengths) else math.nan,
        "average_velocity": float(np.mean(finite_velocities)) if len(finite_velocities) else math.nan,
        "average_lap_time_successes": float(np.mean(finite_lap_times)) if len(finite_lap_times) else math.nan,
        "final_gate_index_mean": float(np.mean(gate_indices)) if len(gate_indices) else math.nan,
    }


def resolve_model_class(args: argparse.Namespace, run_config: Dict[str, Any]):
    if args.model_class == "mpve":
        from scripts.train_acmpc_gym_mpve import MPVEPPO

        return MPVEPPO
    if args.model_class == "base":
        from stable_baselines3 import PPO

        return PPO

    ppo_config = run_config.get("ppo_config", {})
    if "mpve_coef" in ppo_config or "mpve_horizon" in ppo_config:
        from scripts.train_acmpc_gym_mpve import MPVEPPO

        return MPVEPPO

    from stable_baselines3 import PPO

    return PPO


def main() -> None:
    args = build_arg_parser().parse_args()

    run_dir = args.run_dir
    if run_dir is None and args.latest:
        run_dir = latest_run_dir(args.runs_root)
    if run_dir is not None:
        run_dir = run_dir.resolve()

    run_config = read_run_config(run_dir)
    model_path, vecnormalize_path = select_model_and_vecnormalize(run_dir, args.model_path, args.vecnormalize_path)

    config_args = run_config.get("args", {})
    acmpc_t = args.acmpc_t or config_args.get("acmpc_t") or 2
    os.environ["ACMPC_T"] = str(acmpc_t)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    env_config = make_eval_config(args, run_config)
    if vecnormalize_path is None and not args.allow_missing_vecnormalize:
        normalize_expected = bool(run_config.get("env_config", {}).get("observation", {}).get("normalize", False))
        if normalize_expected:
            raise FileNotFoundError(
                "VecNormalize statistics are expected by this run but were not found. "
                "Pass --allow-missing-vecnormalize only for debugging raw-observation evaluation."
            )
    env = make_eval_env(env_config, vecnormalize_path)

    from training_modules.mlp_mpc_policy import MlpMpcPolicy

    del MlpMpcPolicy  # Imported so model loading can resolve the custom policy class.
    model_cls = resolve_model_class(args, run_config)
    model = model_cls.load(str(model_path), env=env, device=args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif run_dir is not None:
        output_dir = run_dir / "eval" / timestamp
    else:
        output_dir = REPO_ROOT / "runs" / "acmpc_gym_eval" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    track = get_track_from_env(env)
    (output_dir / "track.json").write_text(
        json.dumps(serialize_track(track), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metadata = {
        "run_dir": str(run_dir) if run_dir is not None else None,
        "model_path": str(model_path.resolve()),
        "vecnormalize_path": str(vecnormalize_path.resolve()) if vecnormalize_path is not None else None,
        "acmpc_t": int(acmpc_t),
        "deterministic": not args.stochastic,
        "seed": args.seed,
        "env_config": to_jsonable(env_config),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    dt = float(env_config.dynamics.dt)
    max_episode_steps = int(env_config.max_episode_steps)
    summaries: List[Dict[str, Any]] = []
    deterministic = not args.stochastic
    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        summary, rows = run_one_episode(
            env=env,
            model=model,
            episode_index=episode,
            seed=episode_seed,
            deterministic=deterministic,
            max_episode_steps=max_episode_steps,
            dt=dt,
        )
        summaries.append(summary)
        write_trajectory_csv(trajectory_dir / f"trajectory_episode_{episode:04d}.csv", rows)
        status = "success" if summary["success"] else "collision" if summary["collision"] else "timeout"
        print(
            f"episode={episode:04d} seed={episode_seed} status={status} "
            f"return={summary['return']:.3f} length={summary['length']} "
            f"gate={summary['final_gate_index']}"
        )

    metrics = aggregate_metrics(summaries)
    write_summary_csv(output_dir / "summary.csv", summaries)
    (output_dir / "summary.json").write_text(
        json.dumps({"episodes": summaries, "metrics": metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    env.close()

    print("Evaluation summary:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6g}")
        else:
            print(f"  {key}: {value}")
    print(f"Saved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
