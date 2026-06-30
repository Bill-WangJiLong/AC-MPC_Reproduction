"""Track/config helpers for the modified Flightmare racing environment."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if value.lower() in {"yes", "true"}:
        return True
    if value.lower() in {"no", "false"}:
        return False
    if value.startswith("["):
        return ast.literal_eval(value)
    try:
        if any(token in value.lower() for token in (".", "e")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def _split_key_value(line: str) -> tuple[str, Any]:
    key, value = line.split(":", 1)
    return key.strip(), _parse_scalar(value)


def _load_yaml_if_available(path: Path) -> Optional[Dict[str, Any]]:
    try:
        import yaml  # type: ignore
    except Exception:
        return None

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else None


def _top_level_sections(lines: List[str]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw_line.startswith((" ", "\t")) and stripped.endswith(":"):
            current = stripped[:-1]
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(raw_line.rstrip())
    return sections


def _parse_simple_section(lines: List[str]) -> Dict[str, Any]:
    section: Dict[str, Any] = {}
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        if stripped.endswith(":") or stripped.startswith("- "):
            continue
        key, value = _split_key_value(stripped)
        section[key] = value
    return section


def _parse_track_section(lines: List[str]) -> Dict[str, Any]:
    track: Dict[str, Any] = {"start": {}, "finish": {}, "gates": []}
    current_gate: Optional[Dict[str, Any]] = None
    in_start = False
    in_finish = False
    in_gates = False

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped == "start:":
            in_start = True
            in_finish = False
            in_gates = False
            continue
        if stripped == "finish:":
            in_start = False
            in_finish = True
            in_gates = False
            continue
        if stripped == "gates:":
            in_start = False
            in_finish = False
            in_gates = True
            continue

        if in_gates and stripped.startswith("- "):
            if current_gate is not None:
                track["gates"].append(current_gate)
            current_gate = {}
            rest = stripped[2:].strip()
            if rest:
                key, value = _split_key_value(rest)
                current_gate[key] = value
            continue

        if ":" not in stripped:
            continue
        key, value = _split_key_value(stripped)
        if in_start:
            track["start"][key] = value
        elif in_finish:
            track["finish"][key] = value
        elif in_gates and current_gate is not None:
            current_gate[key] = value
        else:
            track[key] = value

    if current_gate is not None:
        track["gates"].append(current_gate)
    return track


def _load_minimal_yaml(path: Path) -> Dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    sections = _top_level_sections(lines)
    data: Dict[str, Any] = {}
    for name, section_lines in sections.items():
        if name == "track":
            data[name] = _parse_track_section(section_lines)
        else:
            data[name] = _parse_simple_section(section_lines)
    return data


def load_racing_yaml(path: Path) -> Dict[str, Any]:
    """Load the Flightmare racing YAML with PyYAML if available, else a small local parser."""

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Racing config does not exist: {path}")

    data = _load_yaml_if_available(path)
    if data is None:
        data = _load_minimal_yaml(path)

    track = data.get("track", {})
    track_path = track.get("track_path") if isinstance(track, dict) else None
    if track_path:
        nested_path = Path(track_path)
        if not nested_path.is_absolute():
            nested_path = path.parent / nested_path
        nested_data = load_racing_yaml(nested_path)
        data["track"] = nested_data.get("track", {})
    return data


def _unit(vector: Any, fallback: List[float]) -> np.ndarray:
    result = np.asarray(vector if vector is not None else fallback, dtype=np.float64)
    norm = np.linalg.norm(result)
    if not math.isfinite(norm) or norm < 1e-9:
        result = np.asarray(fallback, dtype=np.float64)
        norm = np.linalg.norm(result)
    return result / norm


def _gate_to_serializable(index: int, gate: Dict[str, Any]) -> Dict[str, Any]:
    center = np.asarray(gate.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
    normal = _unit(gate.get("normal"), [1.0, 0.0, 0.0])
    up = _unit(gate.get("up"), [0.0, 0.0, 1.0])
    right = np.cross(up, normal)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        right = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        right = right / right_norm
    width = float(gate.get("width", 1.5))
    height = float(gate.get("height", 1.5))
    half_w = 0.5 * width
    half_h = 0.5 * height
    corners = np.stack(
        [
            center - half_w * right - half_h * up,
            center + half_w * right - half_h * up,
            center + half_w * right + half_h * up,
            center - half_w * right + half_h * up,
        ],
        axis=0,
    )
    return {
        "index": index,
        "label": str(gate.get("label", f"G{index}")),
        "center": center.astype(float).tolist(),
        "normal": normal.astype(float).tolist(),
        "up": up.astype(float).tolist(),
        "right": right.astype(float).tolist(),
        "width": width,
        "height": height,
        "frame_thickness": float(gate.get("frame_thickness", 0.12)),
        "corners": corners.astype(float).tolist(),
    }


def load_flightmare_track(racing_config_path: Path) -> Dict[str, Any]:
    """Return a plot/eval friendly track dict from Flightmare's racing_env.yaml."""

    data = load_racing_yaml(racing_config_path)
    track = data.get("track", {})
    start = track.get("start", {}) if isinstance(track, dict) else {}
    finish = track.get("finish", {}) if isinstance(track, dict) else {}
    gates = track.get("gates", []) if isinstance(track, dict) else []
    return {
        "name": str(track.get("name", "flightmare_track")) if isinstance(track, dict) else "flightmare_track",
        "start": {
            "position": list(start.get("position", [0.0, 0.0, 0.0])),
            "yaw": float(start.get("yaw", 0.0)),
        },
        "world_bounds": track.get("world_bounds") if isinstance(track, dict) else None,
        "finish": {
            "position": list(finish.get("position", [0.0, 0.0, 0.0])),
            "radius": float(finish.get("radius", 0.5)),
        },
        "gates": [_gate_to_serializable(index, gate) for index, gate in enumerate(gates)],
    }


def load_racing_metadata(racing_config_path: Path) -> Dict[str, Any]:
    """Load small scalar metadata needed by training/evaluation scripts."""

    data = load_racing_yaml(racing_config_path)
    racing = data.get("racing_env", {})
    dynamics = data.get("quadrotor_dynamics", {})
    action = data.get("action", {})
    track = data.get("track", {})
    return {
        "sim_dt": float(racing.get("sim_dt", 0.02)),
        "max_episode_steps": int(racing.get("max_episode_steps", 500)),
        "max_t": float(racing.get("max_t", 10.0)),
        "random_reset": bool(racing.get("random_reset", True)),
        "mass": float(dynamics.get("mass", 0.752)),
        "action_omega_max": list(action.get("omega_max", [10.0, 10.0, 4.0])),
        "thrust_max_per_motor": float(action.get("thrust_max_per_motor", 8.5)),
        "track_name": str(track.get("name", "flightmare_track")) if isinstance(track, dict) else "flightmare_track",
    }
