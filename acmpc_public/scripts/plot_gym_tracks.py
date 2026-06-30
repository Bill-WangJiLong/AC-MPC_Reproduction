"""Visualize gate locations defined by the AC-MPC Gym track assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acmpc_racing_gym.tracks.loader import ASSET_DIR, load_track


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot AC-MPC Gym track gate positions.")
    parser.add_argument(
        "--track-name",
        action="append",
        default=None,
        help="Track asset name without .json. Can be repeated. Defaults to all assets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "track_visualizations",
        help="Directory for generated figures.",
    )
    parser.add_argument("--show", action="store_true", help="Display figures interactively after saving.")
    return parser


def available_track_names() -> List[str]:
    return sorted(path.stem for path in ASSET_DIR.glob("*.json"))


def draw_gate_projection(ax, gate, axes: tuple[int, int], label: str) -> None:
    corners = gate.corners_world()
    xs = np.r_[corners[:, axes[0]], corners[0, axes[0]]]
    ys = np.r_[corners[:, axes[1]], corners[0, axes[1]]]
    ax.plot(xs, ys, color="#111111", linewidth=1.6)
    ax.scatter([gate.center[axes[0]]], [gate.center[axes[1]]], color="#111111", s=18)
    ax.text(gate.center[axes[0]], gate.center[axes[1]], label, fontsize=9, ha="left", va="bottom")


def draw_normal_projection(ax, gate, axes: tuple[int, int], scale: float = 0.35) -> None:
    start = gate.center
    delta = gate.normal * scale
    ax.arrow(
        start[axes[0]],
        start[axes[1]],
        delta[axes[0]],
        delta[axes[1]],
        head_width=0.06,
        length_includes_head=True,
        color="#d62728",
        alpha=0.85,
    )


def draw_track_line(ax, track, axes: tuple[int, int]) -> None:
    points = np.asarray(
        [track.start.position] + [gate.center for gate in track.gates] + [track.finish.position],
        dtype=float,
    )
    ax.plot(points[:, axes[0]], points[:, axes[1]], color="#1f77b4", linestyle="--", linewidth=1.2, alpha=0.8)


def draw_start_projection(ax, track, axes: tuple[int, int]) -> None:
    start = track.start.position
    ax.scatter([start[axes[0]]], [start[axes[1]]], marker="*", color="#2ca02c", s=90, zorder=5)
    ax.text(start[axes[0]], start[axes[1]], "start", fontsize=9, ha="right", va="top")


def draw_finish_projection(ax, track, axes: tuple[int, int]) -> None:
    position = track.finish.position
    circle = plt.Circle(
        (position[axes[0]], position[axes[1]]),
        track.finish.radius,
        facecolor="#ffbf00",
        edgecolor="#9a6700",
        alpha=0.3,
        linewidth=1.5,
        zorder=4,
    )
    ax.add_patch(circle)
    ax.scatter([position[axes[0]]], [position[axes[1]]], marker="X", color="#9a6700", s=45, zorder=5)
    ax.text(position[axes[0]], position[axes[1]], "finish", fontsize=9, ha="left", va="top")


def draw_bounds_projection(ax, track, axes: tuple[int, int]) -> None:
    if track.world_bounds is None:
        return
    bounds = np.asarray(track.world_bounds, dtype=float)
    x_low, x_high = bounds[axes[0]]
    y_low, y_high = bounds[axes[1]]
    xs = [x_low, x_high, x_high, x_low, x_low]
    ys = [y_low, y_low, y_high, y_high, y_low]
    ax.plot(xs, ys, color="#666666", linewidth=1.0, linestyle=":", alpha=0.8)
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)


def configure_projection_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.35)


def draw_track_3d(ax, track) -> None:
    start = track.start.position
    ax.scatter([start[0]], [start[1]], [start[2]], marker="*", color="#2ca02c", s=90, label="start")

    centers = np.asarray([gate.center for gate in track.gates], dtype=float)
    path = np.vstack([start.reshape(1, 3), centers, track.finish.position.reshape(1, 3)])
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color="#1f77b4", linestyle="--", linewidth=1.2)

    for index, gate in enumerate(track.gates):
        label = gate.label or f"G{index}"
        corners = gate.corners_world()
        closed = np.vstack([corners, corners[0]])
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color="#111111", linewidth=1.4)
        ax.scatter([gate.center[0]], [gate.center[1]], [gate.center[2]], color="#111111", s=16)
        ax.text(gate.center[0], gate.center[1], gate.center[2], label, fontsize=8)
        normal_end = gate.center + 0.35 * gate.normal
        ax.plot(
            [gate.center[0], normal_end[0]],
            [gate.center[1], normal_end[1]],
            [gate.center[2], normal_end[2]],
            color="#d62728",
            linewidth=1.4,
        )

    u = np.linspace(0.0, 2.0 * np.pi, 24)
    v = np.linspace(0.0, np.pi, 12)
    radius = track.finish.radius
    finish = track.finish.position
    x = finish[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = finish[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = finish[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="#d49a00", linewidth=0.5, alpha=0.5)
    ax.scatter([finish[0]], [finish[1]], [finish[2]], marker="X", color="#9a6700", s=45)
    ax.text(finish[0], finish[1], finish[2], "finish", fontsize=8)

    ax.set_title("3D gate layout")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    if track.world_bounds is not None:
        draw_bounds_3d(ax, track)
        bounds = np.asarray(track.world_bounds, dtype=float)
        ax.set_xlim(bounds[0, 0], bounds[0, 1])
        ax.set_ylim(bounds[1, 0], bounds[1, 1])
        ax.set_zlim(bounds[2, 0], bounds[2, 1])
        return
    all_points = np.vstack([path] + [gate.corners_world() for gate in track.gates])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    centers_mid = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    span = max(span, 1.0)
    for setter, center in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centers_mid):
        setter(center - 0.55 * span, center + 0.55 * span)


def draw_bounds_3d(ax, track) -> None:
    bounds = np.asarray(track.world_bounds, dtype=float)
    x0, x1 = bounds[0]
    y0, y1 = bounds[1]
    z0, z1 = bounds[2]
    corners = np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for start, end in edges:
        ax.plot(
            [corners[start, 0], corners[end, 0]],
            [corners[start, 1], corners[end, 1]],
            [corners[start, 2], corners[end, 2]],
            color="#666666",
            linestyle=":",
            linewidth=0.8,
            alpha=0.65,
        )


def print_gate_table(track) -> None:
    print(f"Track: {track.name}")
    print(f"  start: position={track.start.position.tolist()}, yaw={track.start.yaw}")
    print(f"  finish: position={track.finish.position.tolist()}, radius={track.finish.radius}")
    if track.world_bounds is not None:
        print(f"  world_bounds: {np.asarray(track.world_bounds).tolist()}")
    for index, gate in enumerate(track.gates):
        label = gate.label or f"G{index}"
        print(
            "  "
            f"{label}: center={gate.center.tolist()}, normal={gate.normal.tolist()}, "
            f"width={gate.width}, height={gate.height}"
        )


def plot_single_track(track_name: str, output_dir: Path, show: bool) -> Path:
    track = load_track(track_name)
    print_gate_table(track)

    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    top_ax = fig.add_subplot(1, 3, 1)
    side_ax = fig.add_subplot(1, 3, 2)
    ax3d = fig.add_subplot(1, 3, 3, projection="3d")

    for index, gate in enumerate(track.gates):
        label = gate.label or f"G{index}"
        draw_gate_projection(top_ax, gate, (0, 1), label)
        draw_gate_projection(side_ax, gate, (0, 2), label)
        draw_normal_projection(top_ax, gate, (0, 1))
        draw_normal_projection(side_ax, gate, (0, 2))

    draw_track_line(top_ax, track, (0, 1))
    draw_track_line(side_ax, track, (0, 2))
    draw_start_projection(top_ax, track, (0, 1))
    draw_start_projection(side_ax, track, (0, 2))
    draw_finish_projection(top_ax, track, (0, 1))
    draw_finish_projection(side_ax, track, (0, 2))
    configure_projection_axis(top_ax, "Top view (x-y)", "x [m]", "y [m]")
    configure_projection_axis(side_ax, "Side view (x-z)", "x [m]", "z [m]")
    draw_bounds_projection(top_ax, track, (0, 1))
    draw_bounds_projection(side_ax, track, (0, 2))
    draw_track_3d(ax3d, track)

    fig.suptitle(f"AC-MPC Gym track: {track.name}", fontsize=14)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{track.name}_gates.png"
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def plot_overview(track_names: Iterable[str], output_dir: Path, show: bool) -> Optional[Path]:
    tracks = [load_track(name) for name in track_names]
    if not tracks:
        return None
    fig, axes = plt.subplots(len(tracks), 2, figsize=(11, 4 * len(tracks)), constrained_layout=True)
    if len(tracks) == 1:
        axes = np.asarray([axes])

    for row, track in enumerate(tracks):
        top_ax, side_ax = axes[row]
        for index, gate in enumerate(track.gates):
            label = gate.label or f"G{index}"
            draw_gate_projection(top_ax, gate, (0, 1), label)
            draw_gate_projection(side_ax, gate, (0, 2), label)
        draw_track_line(top_ax, track, (0, 1))
        draw_track_line(side_ax, track, (0, 2))
        draw_start_projection(top_ax, track, (0, 1))
        draw_start_projection(side_ax, track, (0, 2))
        draw_finish_projection(top_ax, track, (0, 1))
        draw_finish_projection(side_ax, track, (0, 2))
        configure_projection_axis(top_ax, f"{track.name}: top view", "x [m]", "y [m]")
        configure_projection_axis(side_ax, f"{track.name}: side view", "x [m]", "z [m]")
        draw_bounds_projection(top_ax, track, (0, 1))
        draw_bounds_projection(side_ax, track, (0, 2))

    output_path = output_dir / "all_tracks_overview.png"
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def main() -> None:
    args = build_arg_parser().parse_args()
    track_names = args.track_name or available_track_names()
    output_dir = args.output_dir.resolve()

    saved_paths = [plot_single_track(name, output_dir, args.show) for name in track_names]
    overview_path = plot_overview(track_names, output_dir, args.show)
    if overview_path is not None:
        saved_paths.append(overview_path)

    print("Saved figures:")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
