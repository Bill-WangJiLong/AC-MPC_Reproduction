"""Evaluate a trained AC-MPC policy in the modified Flightmare RacingEnv_v1."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th


REPO_ROOT = Path(__file__).resolve().parents[1]
DIFF_MPC_DRONES = REPO_ROOT / "diff_mpc_drones"
for path in (REPO_ROOT, DIFF_MPC_DRONES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.eval_acmpc_gym import (  # noqa: E402
    aggregate_metrics,
    latest_run_dir,
    positive_int,
    read_run_config,
    resolve_model_class,
    select_model_and_vecnormalize,
    scalar_info,
    to_jsonable,
    write_summary_csv,
    write_trajectory_csv,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an AC-MPC policy in modified Flightmare RacingEnv_v1.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs" / "acmpc_flightmare")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--vecnormalize-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--flightmare-path", type=Path, default=Path(r"D:\MyProjects\flightmare"))
    parser.add_argument(
        "--racing-config-path",
        type=Path,
        default=None,
        help=(
            "Runtime racing_env.yaml. The current RacingEnv_v1 binary only supports "
            "<flightmare-path>/flightlib/configs/racing_env.yaml."
        ),
    )
    parser.add_argument("--episodes", type=positive_int, default=32)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-class", choices=["auto", "base", "mpve"], default="auto")
    parser.add_argument("--acmpc-t", type=positive_int, default=None)
    parser.add_argument("--max-episode-steps", type=positive_int, default=None)
    parser.add_argument("--allow-missing-vecnormalize", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Do not generate trajectory speed heatmaps.")
    parser.add_argument("--plot-max-episodes", type=positive_int, default=64)
    parser.add_argument("--speed-vmin", type=float, default=0.0, help="Heatmap lower speed bound in m/s.")
    parser.add_argument("--speed-vmax", type=float, default=None, help="Heatmap upper speed bound in m/s.")
    return parser


def make_eval_env(
    args: argparse.Namespace,
    run_config: Dict[str, Any],
    vecnormalize_path: Optional[Path],
    vec_config_dir: Path,
):
    from stable_baselines3.common.vec_env import VecNormalize

    from acmpc_flightmare.vec_env import FlightmareRacingVecEnv, write_vec_env_config

    config_args = run_config.get("args", {})
    flightmare_path = args.flightmare_path
    if "flightmare_path" in config_args and args.flightmare_path == Path(r"D:\MyProjects\flightmare"):
        flightmare_path = Path(config_args["flightmare_path"])
    racing_config_path = args.racing_config_path
    if racing_config_path is None:
        racing_from_config = run_config.get("env_config", {}).get("racing_config_path")
        if racing_from_config:
            racing_config_path = Path(racing_from_config)
        else:
            racing_config_path = flightmare_path / "flightlib" / "configs" / "racing_env.yaml"

    vec_config_path = write_vec_env_config(
        vec_config_dir / "eval_vec_env.yaml",
        seed=args.seed,
        num_envs=1,
        num_threads=1,
        render=False,
        scene_id=int(config_args.get("scene_id", 0)),
    )
    raw_env = FlightmareRacingVecEnv(
        flightmare_path=flightmare_path.resolve(),
        vec_config_path=vec_config_path,
        racing_config_path=racing_config_path.resolve(),
    )
    if vecnormalize_path is None:
        return raw_env, raw_env
    env = VecNormalize.load(str(vecnormalize_path), raw_env)
    env.training = False
    env.norm_reward = False
    env.state_space = raw_env.state_space
    return env, raw_env


def run_one_episode(
    env: Any,
    model: Any,
    episode_index: int,
    seed: int,
    deterministic: bool,
    max_episode_steps: int,
    dt: float,
    mass: float,
    thrust_max_per_motor: float,
    omega_cmd_max: np.ndarray,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    env.seed(seed)
    obs = env.reset()
    rows: List[Dict[str, Any]] = []
    total_reward = 0.0
    done = np.array([False])
    info: Dict[str, Any] = {}

    initial_state = env.get_state()[0]
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
            "speed": float(np.linalg.norm(initial_state[7:10])),
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
        speed = float(info.get("speed", np.linalg.norm(velocity)))
        action0 = action.reshape(-1)
        force_mean = (thrust_max_per_motor * 4.0 / mass) / 2.0
        mass_normalized_thrust = (float(action0[0]) + 1.0) * force_mean
        collective_thrust_n = mass * mass_normalized_thrust
        omega_command = action0[1:4] * omega_cmd_max

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
                "mass_normalized_thrust": mass_normalized_thrust,
                "collective_thrust_N": collective_thrust_n,
                "cmd_wx": float(omega_command[0]),
                "cmd_wy": float(omega_command[1]),
                "cmd_wz": float(omega_command[2]),
            }
        )
        if bool(done[0]):
            break

    last_row = rows[-1]
    speed_values = [row["speed"] for row in rows[1:] if np.isfinite(row["speed"])]
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


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = args.run_dir
    if run_dir is None and args.latest:
        run_dir = latest_run_dir(args.runs_root)
    if run_dir is not None:
        run_dir = run_dir.resolve()

    run_config = read_run_config(run_dir)
    model_path, vecnormalize_path = select_model_and_vecnormalize(run_dir, args.model_path, args.vecnormalize_path)
    if vecnormalize_path is None and not args.allow_missing_vecnormalize:
        normalize_expected = bool(run_config.get("env_config", {}).get("normalize_obs", False))
        if normalize_expected:
            raise FileNotFoundError(
                "VecNormalize statistics are expected by this Flightmare run but were not found. "
                "Pass --allow-missing-vecnormalize only for debugging raw-observation evaluation."
            )

    config_args = run_config.get("args", {})
    acmpc_t = args.acmpc_t or config_args.get("acmpc_t") or 2
    os.environ["ACMPC_T"] = str(acmpc_t)
    os.environ["FLIGHTMARE_PATH"] = str(args.flightmare_path.resolve())
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif run_dir is not None:
        output_dir = run_dir / "eval" / timestamp
    else:
        output_dir = REPO_ROOT / "runs" / "acmpc_flightmare_eval" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    env, raw_env = make_eval_env(args, run_config, vecnormalize_path, output_dir)

    from training_modules.mlp_mpc_policy import MlpMpcPolicy

    del MlpMpcPolicy
    model_cls = resolve_model_class(args, run_config)
    model = model_cls.load(str(model_path), env=env, device=args.device)

    (output_dir / "track.json").write_text(
        json.dumps(raw_env.track, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metadata = {
        "run_dir": str(run_dir) if run_dir is not None else None,
        "model_path": str(model_path.resolve()),
        "vecnormalize_path": str(vecnormalize_path.resolve()) if vecnormalize_path is not None else None,
        "flightmare_path": str(raw_env.flightmare_path),
        "racing_config_path": str(raw_env.racing_config_path),
        "acmpc_t": int(acmpc_t),
        "deterministic": not args.stochastic,
        "seed": args.seed,
        "env_metadata": to_jsonable(raw_env.metadata_dict),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    dt = float(raw_env.dt)
    max_episode_steps = int(args.max_episode_steps or raw_env.max_episode_steps)
    mass = float(raw_env.metadata_dict["mass"])
    thrust_max_per_motor = float(raw_env.metadata_dict["thrust_max_per_motor"])
    omega_cmd_max = np.asarray(raw_env.metadata_dict["action_omega_max"], dtype=np.float32)
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
            mass=mass,
            thrust_max_per_motor=thrust_max_per_motor,
            omega_cmd_max=omega_cmd_max,
        )
        summaries.append(summary)
        write_trajectory_csv(trajectory_dir / f"trajectory_episode_{episode:04d}.csv", rows)
        status = "success" if summary["success"] else "collision" if summary["collision"] else "timeout"
        print(
            f"episode={episode:04d} seed={episode_seed} status={status} "
            f"return={summary['return']:.3f} length={summary['length']} gate={summary['final_gate_index']}"
        )

    metrics = aggregate_metrics(summaries)
    write_summary_csv(output_dir / "summary.csv", summaries)
    (output_dir / "summary.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    env.close()

    if not args.no_plots:
        from scripts.plot_trajectories import plot_trajectories

        plot_trajectories(
            eval_dir=output_dir,
            output_dir=output_dir / "plots",
            max_episodes=min(args.plot_max_episodes, args.episodes),
            speed_vmin=args.speed_vmin,
            speed_vmax=args.speed_vmax,
            show=False,
        )

    print("Evaluation summary:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"Saved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
