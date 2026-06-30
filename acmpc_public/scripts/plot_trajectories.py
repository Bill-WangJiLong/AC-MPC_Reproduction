"""Plot AC-MPC Gym evaluation trajectories and gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot trajectories saved by eval_acmpc_gym.py.")
    parser.add_argument("--eval-dir", type=Path, required=True, help="Evaluation output directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for plot files.")
    parser.add_argument("--max-episodes", type=positive_int, default=64)
    parser.add_argument("--speed-vmin", type=float, default=0.0, help="Colorbar lower speed bound in m/s.")
    parser.add_argument("--speed-vmax", type=float, default=None, help="Colorbar upper speed bound in m/s.")
    parser.add_argument("--show", action="store_true")
    return parser


def load_track(eval_dir: Path) -> Dict:
    track_path = eval_dir / "track.json"
    if not track_path.exists():
        raise FileNotFoundError(f"track.json does not exist: {track_path}")
    return json.loads(track_path.read_text(encoding="utf-8"))


def load_metadata(eval_dir: Path) -> Dict:
    metadata_path = eval_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_summary(eval_dir: Path) -> Dict[int, Dict]:
    summary_path = eval_dir / "summary.csv"
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    return {int(row["episode"]): row.to_dict() for _, row in summary.iterrows()}


def trajectory_files(eval_dir: Path, max_episodes: int) -> Iterable[Path]:
    files = sorted((eval_dir / "trajectories").glob("trajectory_episode_*.csv"))
    return files[:max_episodes]


def load_trajectories(eval_dir: Path, max_episodes: int) -> List[pd.DataFrame]:
    trajectories = []
    for path in trajectory_files(eval_dir, max_episodes):
        df = pd.read_csv(path)
        if not df.empty:
            if "speed" not in df.columns:
                df["speed"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2 + df["vz"] ** 2)
            trajectories.append(df)
    if not trajectories:
        raise FileNotFoundError(f"No trajectory CSV files found under: {eval_dir / 'trajectories'}")
    return trajectories


def episode_status(df: pd.DataFrame, summary_by_episode: Dict[int, Dict]) -> str:
    episode = int(df["episode"].iloc[0])
    summary = summary_by_episode.get(episode)
    if summary is not None:
        if bool(summary.get("success", False)):
            return "success"
        if bool(summary.get("collision", False)):
            return "collision"
        return "timeout"
    last = df.iloc[-1]
    if bool(last.get("finished", False)):
        return "success"
    if bool(last.get("collision", False)):
        return "collision"
    return "timeout"


def status_color(status: str) -> str:
    if status == "success":
        return "#1f77b4"
    if status == "collision":
        return "#d62728"
    return "#ff7f0e"


def status_marker(status: str) -> str:
    if status == "success":
        return "o"
    if status == "collision":
        return "x"
    return "s"


def draw_gate_projection(ax, gate: Dict, axes: tuple[str, str], label: Optional[str] = None) -> None:
    axis_index = {"x": 0, "y": 1, "z": 2}
    i0 = axis_index[axes[0]]
    i1 = axis_index[axes[1]]
    corners = gate["corners"]
    xs = [corner[i0] for corner in corners] + [corners[0][i0]]
    ys = [corner[i1] for corner in corners] + [corners[0][i1]]
    ax.plot(xs, ys, color="#222222", linewidth=1.4)
    center = gate["center"]
    ax.scatter([center[i0]], [center[i1]], color="#222222", s=14)
    if label is not None:
        ax.text(center[i0], center[i1], label, fontsize=8, ha="left", va="bottom")


def draw_finish_projection(ax, finish: Dict, axes: tuple[str, str]) -> None:
    axis_index = {"x": 0, "y": 1, "z": 2}
    i0 = axis_index[axes[0]]
    i1 = axis_index[axes[1]]
    position = np.asarray(finish["position"], dtype=float)
    radius = float(finish["radius"])
    circle = plt.Circle(
        (position[i0], position[i1]),
        radius,
        facecolor="#ffbf00",
        edgecolor="#9a6700",
        alpha=0.3,
        linewidth=1.4,
        zorder=3,
    )
    ax.add_patch(circle)
    ax.scatter([position[i0]], [position[i1]], marker="X", color="#9a6700", s=40, zorder=4)


def speed_limits(trajectories: List[pd.DataFrame], speed_vmin: Optional[float], speed_vmax: Optional[float]) -> tuple[float, float]:
    speeds = np.concatenate([df["speed"].to_numpy(dtype=float) for df in trajectories])
    speeds = speeds[np.isfinite(speeds)]
    if speeds.size == 0:
        return 0.0, 1.0
    vmin = float(speed_vmin) if speed_vmin is not None else float(np.min(speeds))
    vmax = float(speed_vmax) if speed_vmax is not None else float(np.max(speeds))
    if vmax <= vmin + 1e-9:
        vmax = vmin + 1.0
    return vmin, vmax


def add_speed_colored_trajectory(ax, df: pd.DataFrame, axes: tuple[str, str], norm: Normalize, cmap: str) -> None:
    x_name, y_name = axes
    coords = df[[x_name, y_name]].to_numpy(dtype=float)
    speeds = df["speed"].to_numpy(dtype=float)
    finite = np.isfinite(coords).all(axis=1) & np.isfinite(speeds)
    coords = coords[finite]
    speeds = speeds[finite]
    if coords.shape[0] < 2:
        return

    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    segment_speeds = 0.5 * (speeds[:-1] + speeds[1:])
    collection = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.5, alpha=0.9)
    collection.set_array(segment_speeds)
    ax.add_collection(collection)
    ax.update_datalim(coords)
    ax.autoscale_view()


def add_status_endpoint(ax, df: pd.DataFrame, axes: tuple[str, str], status: str) -> None:
    x_name, y_name = axes
    last = df.iloc[-1]
    marker = status_marker(status)
    if marker == "x":
        ax.scatter(
            [last[x_name]],
            [last[y_name]],
            marker=marker,
            color="#111111",
            linewidths=0.9,
            s=32,
            zorder=5,
        )
        return
    ax.scatter(
        [last[x_name]],
        [last[y_name]],
        marker=marker,
        facecolors="none",
        edgecolors="#111111",
        linewidths=0.9,
        s=32,
        zorder=5,
    )


def add_physical_control_columns(trajectories: List[pd.DataFrame], metadata: Dict) -> None:
    env_metadata = metadata.get("env_metadata", {})
    mass = float(env_metadata.get("mass", 0.752))
    thrust_max_per_motor = float(env_metadata.get("thrust_max_per_motor", 8.5))
    omega_max = np.asarray(env_metadata.get("action_omega_max", [10.0, 10.0, 4.0]), dtype=float)
    force_mean = (4.0 * thrust_max_per_motor / mass) / 2.0

    for df in trajectories:
        df["mass_normalized_thrust"] = (df["action_thrust"] + 1.0) * force_mean
        df["collective_thrust_N"] = mass * df["mass_normalized_thrust"]
        for action_name, command_name, limit in zip(
            ("action_wx", "action_wy", "action_wz"),
            ("cmd_wx", "cmd_wy", "cmd_wz"),
            omega_max,
        ):
            df[command_name] = df[action_name] * limit


def mean_and_std_by_time(trajectories: List[pd.DataFrame], columns: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    controls = pd.concat([df[["time_s", *columns]] for df in trajectories], ignore_index=True)
    controls = controls.dropna(subset=columns, how="all")
    grouped = controls.groupby("time_s", sort=True)[columns]
    return grouped.mean(), grouped.std(ddof=0).fillna(0.0)


def plot_mean_control_inputs(
    trajectories: List[pd.DataFrame],
    metadata: Dict,
    output_dir: Path,
) -> Path:
    add_physical_control_columns(trajectories, metadata)
    normalized_columns = ["action_thrust", "action_wx", "action_wy", "action_wz"]
    physical_columns = ["mass_normalized_thrust", "cmd_wx", "cmd_wy", "cmd_wz"]
    norm_mean, norm_std = mean_and_std_by_time(trajectories, normalized_columns)
    physical_mean, physical_std = mean_and_std_by_time(trajectories, physical_columns)

    env_metadata = metadata.get("env_metadata", {})
    mass = float(env_metadata.get("mass", 0.752))
    thrust_max_per_motor = float(env_metadata.get("thrust_max_per_motor", 8.5))
    force_mean = (4.0 * thrust_max_per_motor / mass) / 2.0
    hover_action = 9.81 / force_mean - 1.0

    colors = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    thrust_norm_ax, thrust_physical_ax, rates_norm_ax, rates_physical_ax = axes.ravel()

    time = norm_mean.index.to_numpy(dtype=float)
    thrust_mean = norm_mean["action_thrust"].to_numpy(dtype=float)
    thrust_std = norm_std["action_thrust"].to_numpy(dtype=float)
    thrust_norm_ax.plot(time, thrust_mean, color="#222222", label="mean $a_0$")
    thrust_norm_ax.fill_between(time, thrust_mean - thrust_std, thrust_mean + thrust_std, color="#777777", alpha=0.25, label="mean +/- 1 std")
    thrust_norm_ax.axhline(-1.0, color="#555555", linestyle=":", label="zero thrust")
    thrust_norm_ax.axhline(hover_action, color="#d62728", linestyle="--", label="hover equivalent")
    thrust_norm_ax.set_title("Normalized collective thrust")
    thrust_norm_ax.set_ylabel("normalized action")
    thrust_norm_ax.set_ylim(-1.05, 1.05)
    thrust_norm_ax.legend(fontsize=8)

    physical_time = physical_mean.index.to_numpy(dtype=float)
    physical_thrust = physical_mean["mass_normalized_thrust"].to_numpy(dtype=float)
    physical_thrust_std = physical_std["mass_normalized_thrust"].to_numpy(dtype=float)
    thrust_physical_ax.plot(physical_time, physical_thrust, color="#222222", label="mean $c$")
    thrust_physical_ax.fill_between(
        physical_time,
        physical_thrust - physical_thrust_std,
        physical_thrust + physical_thrust_std,
        color="#777777",
        alpha=0.25,
        label="mean +/- 1 std",
    )
    thrust_physical_ax.axhline(9.81, color="#d62728", linestyle="--", label="$g=9.81$ m/s^2")
    thrust_physical_ax.axhline(0.0, color="#555555", linestyle=":")
    thrust_physical_ax.set_title("Physical collective thrust command")
    thrust_physical_ax.set_ylabel("mass-normalized thrust [m/s^2]")
    thrust_physical_ax.legend(fontsize=8)

    for axis_name, column in zip(("x", "y", "z"), normalized_columns[1:]):
        values = norm_mean[column].to_numpy(dtype=float)
        spread = norm_std[column].to_numpy(dtype=float)
        rates_norm_ax.plot(time, values, color=colors[axis_name], label=f"$a_{{\\omega_{axis_name}}}$")
        rates_norm_ax.fill_between(time, values - spread, values + spread, color=colors[axis_name], alpha=0.15)
    rates_norm_ax.set_title("Normalized body-rate actions")
    rates_norm_ax.set_ylabel("normalized action")
    rates_norm_ax.set_ylim(-1.05, 1.05)
    rates_norm_ax.legend(fontsize=8, ncols=3)

    for axis_name, column in zip(("x", "y", "z"), physical_columns[1:]):
        values = physical_mean[column].to_numpy(dtype=float)
        spread = physical_std[column].to_numpy(dtype=float)
        rates_physical_ax.plot(physical_time, values, color=colors[axis_name], label=f"$\\omega_{axis_name}$")
        rates_physical_ax.fill_between(physical_time, values - spread, values + spread, color=colors[axis_name], alpha=0.15)
    rates_physical_ax.set_title("Physical body-rate commands")
    rates_physical_ax.set_ylabel("body rate [rad/s]")
    rates_physical_ax.legend(fontsize=8, ncols=3)

    for ax in axes.ravel():
        ax.set_xlabel("time [s]")
        ax.grid(True, linewidth=0.4, alpha=0.35)

    output_path = output_dir / "mean_control_inputs.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_trajectories(
    eval_dir: Path,
    output_dir: Path,
    max_episodes: int,
    speed_vmin: Optional[float],
    speed_vmax: Optional[float],
    show: bool,
) -> None:
    track = load_track(eval_dir)
    metadata = load_metadata(eval_dir)
    summary_by_episode = load_summary(eval_dir)
    trajectories = load_trajectories(eval_dir, max_episodes)
    cmap = "coolwarm"
    norm = Normalize(*speed_limits(trajectories, speed_vmin, speed_vmax))

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    top_ax, side_ax = axes

    for gate in track["gates"]:
        label = gate.get("label", f"G{gate['index']}")
        draw_gate_projection(top_ax, gate, ("x", "y"), label=label)
        draw_gate_projection(side_ax, gate, ("x", "z"), label=label)
    if "finish" in track:
        draw_finish_projection(top_ax, track["finish"], ("x", "y"))
        draw_finish_projection(side_ax, track["finish"], ("x", "z"))

    for df in trajectories:
        status = episode_status(df, summary_by_episode)
        add_speed_colored_trajectory(top_ax, df, ("x", "y"), norm, cmap)
        add_speed_colored_trajectory(side_ax, df, ("x", "z"), norm, cmap)
        add_status_endpoint(top_ax, df, ("x", "y"), status)
        add_status_endpoint(side_ax, df, ("x", "z"), status)

    top_ax.set_title("Top view")
    top_ax.set_xlabel("x [m]")
    top_ax.set_ylabel("y [m]")
    top_ax.set_aspect("equal", adjustable="box")
    top_ax.grid(True, linewidth=0.4, alpha=0.35)

    side_ax.set_title("Side view")
    side_ax.set_xlabel("x [m]")
    side_ax.set_ylabel("z [m]")
    side_ax.set_aspect("equal", adjustable="box")
    side_ax.grid(True, linewidth=0.4, alpha=0.35)

    handles = [
        plt.Line2D([0], [0], color="#111111", marker="o", markerfacecolor="none", lw=0, label="success end"),
        plt.Line2D([0], [0], color="#111111", marker="x", lw=0, label="collision end"),
        plt.Line2D([0], [0], color="#111111", marker="s", markerfacecolor="none", lw=0, label="timeout end"),
        plt.Line2D([0], [0], color="#222222", lw=2, label="gate"),
        plt.Line2D([0], [0], color="#9a6700", marker="X", lw=0, label="finish region"),
    ]
    fig.legend(handles=handles, loc="upper center", ncols=5, frameon=False)
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    fig.colorbar(scalar_mappable, ax=axes.ravel().tolist(), shrink=0.86, label="speed [m/s]")
    combined_path = output_dir / "trajectories_speed_top_side.png"
    fig.savefig(combined_path, dpi=180)

    top_fig, top_only_ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for gate in track["gates"]:
        label = gate.get("label", f"G{gate['index']}")
        draw_gate_projection(top_only_ax, gate, ("x", "y"), label=label)
    if "finish" in track:
        draw_finish_projection(top_only_ax, track["finish"], ("x", "y"))
    for df in trajectories:
        status = episode_status(df, summary_by_episode)
        add_speed_colored_trajectory(top_only_ax, df, ("x", "y"), norm, cmap)
        add_status_endpoint(top_only_ax, df, ("x", "y"), status)
    top_only_ax.set_title("Top view speed heatmap")
    top_only_ax.set_xlabel("x [m]")
    top_only_ax.set_ylabel("y [m]")
    top_only_ax.set_aspect("equal", adjustable="box")
    top_only_ax.grid(True, linewidth=0.4, alpha=0.35)
    top_scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    top_scalar_mappable.set_array([])
    top_fig.colorbar(top_scalar_mappable, ax=top_only_ax, shrink=0.86, label="speed [m/s]")
    top_path = output_dir / "trajectories_speed_top_view.png"
    top_fig.savefig(top_path, dpi=180)
    controls_path = plot_mean_control_inputs(trajectories, metadata, output_dir)

    if show:
        plt.show()
    else:
        plt.close(fig)
        plt.close(top_fig)

    print("Saved plots:")
    print(f"  {combined_path}")
    print(f"  {top_path}")
    print(f"  {controls_path}")


def main() -> None:
    args = build_arg_parser().parse_args()
    eval_dir = args.eval_dir.resolve()
    output_dir = (args.output_dir or (eval_dir / "plots")).resolve()
    plot_trajectories(eval_dir, output_dir, args.max_episodes, args.speed_vmin, args.speed_vmax, args.show)


if __name__ == "__main__":
    main()
