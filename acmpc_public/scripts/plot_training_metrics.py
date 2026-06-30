"""Plot AC-MPC Gym training diagnostics from a run directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "acmpc_gym"
DEFAULT_PAPER_STEP_SAMPLES = 25_000.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize AC-MPC Gym PPO training logs.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory containing csv/episodes.csv and/or sb3/progress.csv.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help=f"Use the newest directory under {DEFAULT_RUNS_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots. Defaults to <run-dir>/plots.",
    )
    parser.add_argument("--window", type=int, default=50, help="Rolling window over episodes for smoothing.")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--show", action="store_true", help="Open matplotlib windows after saving plots.")
    parser.add_argument(
        "--x-axis",
        choices=("paper-step", "global-timesteps", "updates"),
        default="paper-step",
        help=(
            "Horizontal axis. paper-step matches the paper convention inferred as "
            "environment samples divided by --paper-step-samples."
        ),
    )
    parser.add_argument(
        "--paper-step-samples",
        type=float,
        default=DEFAULT_PAPER_STEP_SAMPLES,
        help="Number of environment samples represented by one paper training Step.",
    )
    return parser


def latest_run_dir(base_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    candidates = [p for p in base_dir.iterdir() if p.is_dir()] if base_dir.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {base_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.latest:
        return latest_run_dir()
    if args.run_dir is None:
        raise ValueError("Pass --run-dir or --latest")
    return args.run_dir.resolve()


def parse_bool_series(series):
    import pandas as pd

    if series.dtype == bool:
        return series.astype(float)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "yes": 1.0, "no": 0.0})
        .fillna(0.0)
    )


def rolling(series, window: int):
    return series.rolling(window=max(int(window), 1), min_periods=1).mean()


def load_csv(path: Path):
    import pandas as pd

    if not path.exists():
        return None
    if path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def _paper_step_label(paper_step_samples: float) -> str:
    if float(paper_step_samples).is_integer():
        samples = f"{int(paper_step_samples):,}"
    else:
        samples = f"{paper_step_samples:g}"
    return f"Step ({samples} env samples)"


def episode_x(episodes, x_axis: str, paper_step_samples: float):
    if x_axis == "paper-step" and "global_timesteps" in episodes:
        return episodes["global_timesteps"] / paper_step_samples, _paper_step_label(paper_step_samples)
    if x_axis == "updates" and "global_timesteps" in episodes:
        return episodes["global_timesteps"] / paper_step_samples, _paper_step_label(paper_step_samples)
    if "global_timesteps" in episodes:
        return episodes["global_timesteps"], "global_timesteps"
    return episodes.index, "episode"


def plot_episode_metrics(
    episodes,
    output_dir: Path,
    window: int,
    dpi: int,
    x_axis: str,
    paper_step_samples: float,
) -> List[Path]:
    import matplotlib.pyplot as plt

    if episodes is None or episodes.empty:
        return []

    x, xlabel = episode_x(episodes, x_axis, paper_step_samples)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    axes = axes.reshape(-1)

    if "episode_return" in episodes:
        axes[0].plot(x, episodes["episode_return"], alpha=0.25, linewidth=0.8, label="episode")
        axes[0].plot(x, rolling(episodes["episode_return"], window), linewidth=2.0, label=f"rolling {window}")
        axes[0].set_title("Reward Evolution")
        axes[0].set_ylabel("episode return")
        axes[0].legend()

    if "finished" in episodes:
        success = parse_bool_series(episodes["finished"])
        axes[1].plot(x, rolling(success, window), linewidth=2.0)
        axes[1].set_title("Success Rate")
        axes[1].set_ylabel("rolling fraction")
        axes[1].set_ylim(-0.05, 1.05)

    if "collision" in episodes:
        collision = parse_bool_series(episodes["collision"])
        axes[2].plot(x, rolling(collision, window), linewidth=2.0, color="tab:red")
        axes[2].set_title("Collision Rate")
        axes[2].set_ylabel("rolling fraction")
        axes[2].set_ylim(-0.05, 1.05)

    if "timeout" in episodes:
        timeout = parse_bool_series(episodes["timeout"])
        axes[3].plot(x, rolling(timeout, window), linewidth=2.0, color="tab:orange")
        axes[3].set_title("Timeout Rate")
        axes[3].set_ylabel("rolling fraction")
        axes[3].set_ylim(-0.05, 1.05)

    if "gate_index" in episodes:
        axes[4].plot(x, episodes["gate_index"], alpha=0.25, linewidth=0.8, label="episode")
        axes[4].plot(x, rolling(episodes["gate_index"], window), linewidth=2.0, label=f"rolling {window}")
        axes[4].set_title("Gate Progress")
        axes[4].set_ylabel("gate_index")
        axes[4].legend()

    if "episode_length" in episodes:
        axes[5].plot(x, episodes["episode_length"], alpha=0.25, linewidth=0.8, label="episode")
        axes[5].plot(x, rolling(episodes["episode_length"], window), linewidth=2.0, label=f"rolling {window}")
        axes[5].set_title("Episode Length")
        axes[5].set_ylabel("steps")
        axes[5].legend()

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(xlabel)

    path = output_dir / "episode_metrics.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return [path]


def first_existing(columns: Iterable[str], frame_columns: Sequence[str]) -> Optional[str]:
    for column in columns:
        if column in frame_columns:
            return column
    return None


def progress_x(progress, x_axis: str, paper_step_samples: float):
    if x_axis == "paper-step":
        for column in ("time/total_timesteps", "total_timesteps"):
            if column in progress:
                return progress[column] / paper_step_samples, _paper_step_label(paper_step_samples)
    if x_axis == "updates":
        if "time/iterations" in progress:
            return progress["time/iterations"], "PPO update"
        return progress.index, "PPO update"
    for column in ("time/total_timesteps", "total_timesteps", "time/iterations"):
        if column in progress:
            return progress[column], column
    return progress.index, "update"


def plot_progress_metrics(
    progress,
    output_dir: Path,
    dpi: int,
    x_axis: str,
    paper_step_samples: float,
) -> List[Path]:
    import matplotlib.pyplot as plt

    if progress is None or progress.empty:
        return []

    x, xlabel = progress_x(progress, x_axis, paper_step_samples)
    figures: List[Tuple[str, List[Tuple[str, str]]]] = [
        (
            "ppo_rollout_metrics.png",
            [
                ("rollout/ep_rew_mean", "ep_rew_mean"),
                ("rollout/ep_len_mean", "ep_len_mean"),
                ("train/std", "policy_std"),
                ("train/learning_rate", "learning_rate"),
            ],
        ),
        (
            "ppo_loss_metrics.png",
            [
                ("train/policy_gradient_loss", "policy_gradient_loss"),
                ("train/value_loss", "value_loss"),
                ("train/entropy_loss", "entropy_loss"),
                ("train/approx_kl", "approx_kl"),
                ("train/clip_fraction", "clip_fraction"),
                ("train/explained_variance", "explained_variance"),
            ],
        ),
        (
            "mpve_metrics.png",
            [
                ("train/td_value_loss", "td_value_loss"),
                ("train/mpve_value_loss", "mpve_value_loss"),
                ("train/mpve_valid_fraction", "mpve_valid_fraction"),
                ("train/value_loss", "combined_value_loss"),
            ],
        ),
    ]

    saved: List[Path] = []
    for filename, specs in figures:
        available = [(column, label) for column, label in specs if column in progress]
        if not available:
            continue

        nrows = (len(available) + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(14, 4.2 * nrows), constrained_layout=True)
        if not isinstance(axes, (list, tuple)):
            axes_list = list(axes.reshape(-1)) if hasattr(axes, "reshape") else [axes]
        else:
            axes_list = list(axes)
        if hasattr(axes, "reshape"):
            axes_list = list(axes.reshape(-1))

        for ax, (column, label) in zip(axes_list, available):
            ax.plot(x, progress[column], linewidth=1.8)
            ax.set_title(label)
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.25)

        for ax in axes_list[len(available) :]:
            ax.axis("off")

        path = output_dir / filename
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved.append(path)

    return saved


def print_summary(episodes, progress, x_axis: str, paper_step_samples: float) -> None:
    print("Plot scale:")
    if x_axis == "paper-step":
        print(f"  x_axis: paper Step = environment samples / {paper_step_samples:g}")
        print("  reward: accumulated episode return or rollout mean episode return; values are not rescaled")
    else:
        print(f"  x_axis: {x_axis}")

    if episodes is not None and not episodes.empty:
        print("Episode CSV summary:")
        print(f"  episodes: {len(episodes)}")
        if "episode_return" in episodes:
            recent = episodes["episode_return"].tail(min(100, len(episodes)))
            print(f"  recent_return_mean: {recent.mean():.4f}")
        if "finished" in episodes:
            success = parse_bool_series(episodes["finished"]).tail(min(100, len(episodes)))
            print(f"  recent_success_rate: {success.mean():.4f}")
        if "collision" in episodes:
            collision = parse_bool_series(episodes["collision"]).tail(min(100, len(episodes)))
            print(f"  recent_collision_rate: {collision.mean():.4f}")
        if "gate_index" in episodes:
            gate = episodes["gate_index"].tail(min(100, len(episodes)))
            print(f"  recent_gate_index_mean: {gate.mean():.4f}")
    else:
        print("Episode CSV summary: csv/episodes.csv not found or empty")

    if progress is not None and not progress.empty:
        print("SB3 progress summary:")
        for column in (
            "rollout/ep_rew_mean",
            "rollout/ep_len_mean",
            "train/td_value_loss",
            "train/mpve_value_loss",
            "train/mpve_valid_fraction",
            "train/value_loss",
            "train/approx_kl",
            "train/std",
        ):
            if column in progress:
                value = progress[column].dropna().iloc[-1] if not progress[column].dropna().empty else None
                if value is not None:
                    print(f"  latest {column}: {value:.6g}")
    else:
        print("SB3 progress summary: sb3/progress.csv not found or empty")


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = resolve_run_dir(args)
    output_dir = (args.output_dir or (run_dir / "plots")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    episodes = load_csv(run_dir / "csv" / "episodes.csv")
    progress = load_csv(run_dir / "sb3" / "progress.csv")

    saved = []
    saved.extend(
        plot_episode_metrics(
            episodes,
            output_dir,
            args.window,
            args.dpi,
            args.x_axis,
            args.paper_step_samples,
        )
    )
    saved.extend(plot_progress_metrics(progress, output_dir, args.dpi, args.x_axis, args.paper_step_samples))

    print(f"Run directory: {run_dir}")
    print(f"Plot directory: {output_dir}")
    print_summary(episodes, progress, args.x_axis, args.paper_step_samples)
    if saved:
        print("Saved plots:")
        for path in saved:
            print(f"  {path}")
    else:
        print("No plots saved. Check whether training logs exist.")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
