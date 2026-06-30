"""Track loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from acmpc_racing_gym.tracks.gate import Gate
from acmpc_racing_gym.tracks.track import FinishRegion, Track, TrackStart


ASSET_DIR = Path(__file__).resolve().parent / "assets"


def load_track(track_name: str = "split_s_like", track_path: Optional[Path] = None) -> Track:
    path = Path(track_path) if track_path is not None else ASSET_DIR / f"{track_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Track file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return _track_from_dict(data)


def _track_from_dict(data: Dict[str, Any]) -> Track:
    gates = [
        Gate(
            center=np.asarray(gate["center"], dtype=np.float64),
            normal=np.asarray(gate["normal"], dtype=np.float64),
            up=np.asarray(gate["up"], dtype=np.float64),
            width=float(gate["width"]),
            height=float(gate["height"]),
            frame_thickness=float(gate.get("frame_thickness", 0.12)),
            label=gate.get("label"),
        )
        for gate in data["gates"]
    ]
    start_data = data.get("start", {})
    start = TrackStart(
        position=np.asarray(start_data.get("position", [-2.0, 0.0, 2.0]), dtype=np.float64),
        yaw=float(start_data.get("yaw", 0.0)),
    )
    finish_data = data.get("finish")
    if not isinstance(finish_data, dict):
        raise ValueError("Track finish section is required")
    finish = FinishRegion(
        position=np.asarray(finish_data["position"], dtype=np.float64),
        radius=float(finish_data["radius"]),
    )
    world_bounds = data.get("world_bounds")
    if world_bounds is not None:
        world_bounds = np.asarray(world_bounds, dtype=np.float32)
        if world_bounds.shape != (3, 2):
            raise ValueError(f"world_bounds must have shape (3, 2), got {world_bounds.shape}")
    return Track(
        name=str(data.get("name", "unnamed")),
        gates=gates,
        start=start,
        finish=finish,
        world_bounds=world_bounds,
    )
