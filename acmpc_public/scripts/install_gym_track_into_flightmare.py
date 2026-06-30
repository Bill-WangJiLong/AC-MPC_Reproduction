"""Install a Python Gym track JSON into Flightmare's default RacingEnv config.

The current Flightmare VecEnv constructs RacingEnv with its default constructor,
so every worker reads:

    <FLIGHTMARE_PATH>/flightlib/configs/racing_env.yaml

This installer converts the Gym JSON to a standalone YAML track file and
updates only the top-level ``track`` section of the default racing config to
reference it. The original racing config is backed up before the first change.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLIGHTMARE_PATH = Path(r"D:\MyProjects\flightmare")
DEFAULT_TRACK_JSON = REPO_ROOT / "acmpc_racing_gym" / "tracks" / "assets" / "split_s.json"


def positive_float(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be a positive finite number, got {value!r}")
    return result


def vector3(value: Any, field: str) -> List[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must be a three-element list")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} contains a non-finite value")
    return result


def world_bounds(value: Any) -> List[List[float]]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("world_bounds must contain x/y/z bounds")
    result: List[List[float]] = []
    for axis, pair in zip(("x", "y", "z"), value):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"world_bounds.{axis} must contain [min, max]")
        low, high = float(pair[0]), float(pair[1])
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError(f"world_bounds.{axis} is invalid: {pair!r}")
        result.append([low, high])
    return result


def validate_track(data: Dict[str, Any]) -> Dict[str, Any]:
    name = str(data.get("name", "")).strip()
    if not name:
        raise ValueError("Track name is required")

    start = data.get("start")
    if not isinstance(start, dict):
        raise ValueError("Track start section is required")
    start_position = vector3(start.get("position"), "start.position")
    start_yaw = float(start.get("yaw", 0.0))
    if not math.isfinite(start_yaw):
        raise ValueError("start.yaw must be finite")

    bounds = world_bounds(data.get("world_bounds"))
    finish = data.get("finish")
    if not isinstance(finish, dict):
        raise ValueError("Track finish section is required")
    finish_position = vector3(finish.get("position"), "finish.position")
    finish_radius = positive_float(finish.get("radius"), "finish.radius")
    gates = data.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("At least one gate is required")

    validated_gates = []
    for index, gate in enumerate(gates):
        if not isinstance(gate, dict):
            raise ValueError(f"gates[{index}] must be an object")
        normal = vector3(gate.get("normal"), f"gates[{index}].normal")
        up = vector3(gate.get("up"), f"gates[{index}].up")
        if sum(item * item for item in normal) < 1e-12:
            raise ValueError(f"gates[{index}].normal must not be zero")
        if sum(item * item for item in up) < 1e-12:
            raise ValueError(f"gates[{index}].up must not be zero")
        validated_gates.append(
            {
                "label": str(gate.get("label", f"G{index}")),
                "center": vector3(gate.get("center"), f"gates[{index}].center"),
                "normal": normal,
                "up": up,
                "width": positive_float(gate.get("width"), f"gates[{index}].width"),
                "height": positive_float(gate.get("height"), f"gates[{index}].height"),
                "frame_thickness": positive_float(
                    gate.get("frame_thickness", 0.12),
                    f"gates[{index}].frame_thickness",
                ),
            }
        )

    return {
        "name": name,
        "world_bounds": bounds,
        "start": {"position": start_position, "yaw": start_yaw},
        "finish": {"position": finish_position, "radius": finish_radius},
        "gates": validated_gates,
    }


def flow(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def render_track_yaml(track: Dict[str, Any]) -> str:
    lines = [
        "track:",
        f"  name: {json.dumps(track['name'], ensure_ascii=False)}",
        "  start:",
        f"    position: {flow(track['start']['position'])}",
        f"    yaw: {track['start']['yaw']:.12g}",
        "  finish:",
        f"    position: {flow(track['finish']['position'])}",
        f"    radius: {track['finish']['radius']:.12g}",
        f"  world_bounds: {flow(track['world_bounds'])}",
        "  gates:",
    ]
    for gate in track["gates"]:
        lines.extend(
            [
                f"    - center: {flow(gate['center'])}",
                f"      normal: {flow(gate['normal'])}",
                f"      up: {flow(gate['up'])}",
                f"      width: {gate['width']:.12g}",
                f"      height: {gate['height']:.12g}",
                f"      frame_thickness: {gate['frame_thickness']:.12g}",
                f"      label: {json.dumps(gate['label'], ensure_ascii=False)}",
            ]
        )
    return "\n".join(lines) + "\n"


def replace_top_level_section(text: str, section: str, replacement: str) -> str:
    lines = text.splitlines(keepends=True)
    start = None
    end = len(lines)
    section_pattern = re.compile(rf"^{re.escape(section)}:\s*(?:#.*)?$")
    top_level_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*:\s*(?:#.*)?$")

    for index, raw_line in enumerate(lines):
        if section_pattern.match(raw_line.strip("\r\n")):
            start = index
            break
    if start is None:
        if text and not text.endswith(("\n", "\r")):
            text += "\n"
        return text + "\n" + replacement

    for index in range(start + 1, len(lines)):
        candidate = lines[index].strip("\r\n")
        if candidate and not candidate[0].isspace() and top_level_pattern.match(candidate):
            end = index
            break

    replacement_lines = replacement.splitlines(keepends=True)
    if replacement and not replacement.endswith("\n"):
        replacement_lines[-1] += "\n"
    return "".join(lines[:start] + replacement_lines + lines[end:])


def install_track(
    flightmare_path: Path,
    track_json: Path,
    generated_track_yaml: Path,
    racing_config: Path,
    backup_path: Path,
    dry_run: bool,
) -> None:
    if not track_json.exists():
        raise FileNotFoundError(f"Gym track JSON does not exist: {track_json}")
    if not racing_config.exists():
        raise FileNotFoundError(f"Flightmare racing config does not exist: {racing_config}")

    data = json.loads(track_json.read_text(encoding="utf-8"))
    track = validate_track(data)
    track_yaml = render_track_yaml(track)
    absolute_track_path = generated_track_yaml.resolve().as_posix()
    track_reference = (
        "track:\n"
        f"  track_path: {json.dumps(absolute_track_path, ensure_ascii=False)}\n"
    )
    updated_config = replace_top_level_section(
        racing_config.read_text(encoding="utf-8"),
        "track",
        track_reference,
    )

    print(f"Gym track: {track_json.resolve()}")
    print(f"Track name: {track['name']}")
    print(f"Gate count: {len(track['gates'])}")
    print(f"Generated Flightmare track: {generated_track_yaml.resolve()}")
    print(f"Flightmare runtime config: {racing_config.resolve()}")
    print(f"Backup: {backup_path.resolve()}")

    if dry_run:
        print("Dry run only; no files were changed.")
        return

    generated_track_yaml.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(racing_config, backup_path)
        print("Created original config backup.")
    else:
        print("Original config backup already exists; it was not overwritten.")

    generated_track_yaml.write_text(track_yaml, encoding="utf-8")
    racing_config.write_text(updated_config, encoding="utf-8")
    print("Installed track into Flightmare runtime configuration.")


def restore_config(racing_config: Path, backup_path: Path) -> None:
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup does not exist: {backup_path}")
    shutil.copy2(backup_path, racing_config)
    print(f"Restored Flightmare racing config from: {backup_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install a Gym JSON track into modified Flightmare RacingEnv.")
    parser.add_argument("--flightmare-path", type=Path, default=DEFAULT_FLIGHTMARE_PATH)
    parser.add_argument("--track-json", type=Path, default=DEFAULT_TRACK_JSON)
    parser.add_argument("--generated-track-yaml", type=Path, default=None)
    parser.add_argument("--racing-config", type=Path, default=None)
    parser.add_argument("--backup-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    flightmare_path = args.flightmare_path.resolve()
    track_json = args.track_json.resolve()
    track_name = track_json.stem
    generated_track_yaml = (
        args.generated_track_yaml.resolve()
        if args.generated_track_yaml is not None
        else flightmare_path / "flightlib" / "configs" / "tracks" / f"{track_name}.yaml"
    )
    racing_config = (
        args.racing_config.resolve()
        if args.racing_config is not None
        else flightmare_path / "flightlib" / "configs" / "racing_env.yaml"
    )
    backup_path = (
        args.backup_path.resolve()
        if args.backup_path is not None
        else racing_config.with_name(racing_config.name + ".pre_track_install.bak")
    )

    if args.restore:
        restore_config(racing_config, backup_path)
        return

    install_track(
        flightmare_path=flightmare_path,
        track_json=track_json,
        generated_track_yaml=generated_track_yaml,
        racing_config=racing_config,
        backup_path=backup_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
