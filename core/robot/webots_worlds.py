"""Utilities for loading Webots world metadata directly from `.wbt` files."""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORLD_DIR = PROJECT_ROOT / "webots" / "worlds"

# Cache parsed world configs so repeated lookups stay cheap.
WORLD_CONFIGS: Dict[str, Dict[str, Any]] = {}

_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_DEFAULT_GOAL_RADIUS = 1.0
_GOAL_COLOR_MAP = {
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}


def _available_world_files() -> List[Path]:
    if not WORLD_DIR.exists():
        return []
    return sorted(WORLD_DIR.rglob("*.wbt"))


def _relative_world_name(path: Path) -> str:
    try:
        return path.relative_to(WORLD_DIR).with_suffix("").as_posix()
    except ValueError:
        return path.with_suffix("").as_posix()


def list_available_worlds() -> list:
    """Return every `.wbt` world available under `webots/worlds`."""
    return [_relative_world_name(path) for path in _available_world_files()]


def _resolve_world_path(world_name: str) -> Path:
    candidate = Path(world_name)

    path_candidates = []
    if candidate.is_absolute():
        path_candidates.append(candidate)
    else:
        path_candidates.extend(
            [
                candidate,
                candidate.with_suffix(".wbt"),
                WORLD_DIR / candidate,
                WORLD_DIR / candidate.with_suffix(".wbt"),
            ]
        )

    for path in path_candidates:
        if path.is_file():
            return path.resolve()

    if not candidate.is_absolute():
        stem = candidate.stem
        matches = [path for path in _available_world_files() if path.stem == stem]
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            options = [_relative_world_name(path) for path in matches]
            raise ValueError(
                f"Ambiguous world name: {world_name}. Use one of: {options}"
            )

    raise ValueError(
        f"Unknown world: {world_name}. Available worlds: {list_available_worlds()}"
    )


def _parse_numbers(text: str) -> List[float]:
    return [float(value) for value in re.findall(_NUMBER_RE, text)]


def _extract_braced_block(text: str, open_brace_index: int) -> Tuple[str, int]:
    depth = 0
    for idx in range(open_brace_index, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_index : idx + 1], idx + 1
    raise ValueError("Unbalanced braces while parsing WBT file")


def _axis_angle_to_matrix(axis: Tuple[float, float, float], angle: float) -> np.ndarray:
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm == 0.0 or abs(angle) < 1e-12:
        return np.eye(3, dtype=np.float64)

    ux, uy, uz = ax / norm, ay / norm, az / norm
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c

    return np.array(
        [
            [t * ux * ux + c, t * ux * uy - s * uz, t * ux * uz + s * uy],
            [t * ux * uy + s * uz, t * uy * uy + c, t * uy * uz - s * ux],
            [t * ux * uz - s * uy, t * uy * uz + s * ux, t * uz * uz + c],
        ],
        dtype=np.float64,
    )


def _normalize_2d(vector: np.ndarray) -> Optional[List[float]]:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return None
    return [float(vector[0] / norm), float(vector[1] / norm)]


def _points_in_polygon(x, z, corners: List[List[float]]):
    x_arr, z_arr = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
    )
    polygon = np.asarray(corners, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        return np.zeros_like(x_arr, dtype=bool)

    inside = np.zeros_like(x_arr, dtype=bool)
    on_edge = np.zeros_like(x_arr, dtype=bool)
    eps = 1e-9

    x_prev, z_prev = polygon[-1]
    for x_curr, z_curr in polygon:
        dx = x_curr - x_prev
        dz = z_curr - z_prev
        cross = (x_arr - x_prev) * dz - (z_arr - z_prev) * dx
        dot = (x_arr - x_prev) * dx + (z_arr - z_prev) * dz
        segment_len_sq = dx * dx + dz * dz
        on_edge |= (np.abs(cross) <= eps) & (dot >= -eps) & (dot <= segment_len_sq + eps)

        intersects = ((z_curr > z_arr) != (z_prev > z_arr)) & (
            x_arr <= (x_prev - x_curr) * (z_arr - z_curr) / (z_prev - z_curr + eps) + x_curr
        )
        inside ^= intersects
        x_prev, z_prev = x_curr, z_curr

    return inside | on_edge


def expanded_obstacle_corners(obstacle: Dict[str, Any], clearance: float = 0.0) -> List[List[float]]:
    """Return obstacle footprint corners, preserving orientation when available."""
    center = obstacle.get("center")
    axes = obstacle.get("axes")
    half_extents = obstacle.get("half_extents")
    if center is not None and axes is not None and half_extents is not None:
        c = np.asarray(center, dtype=np.float64)
        u = np.asarray(axes[0], dtype=np.float64)
        v = np.asarray(axes[1], dtype=np.float64)
        hx = float(half_extents[0]) + float(clearance)
        hz = float(half_extents[1]) + float(clearance)
        corners = np.array(
            [
                c - u * hx - v * hz,
                c + u * hx - v * hz,
                c + u * hx + v * hz,
                c - u * hx + v * hz,
            ],
            dtype=np.float64,
        )
        return corners.tolist()

    if float(clearance) == 0.0 and obstacle.get("corners") is not None:
        return copy.deepcopy(obstacle["corners"])

    (min_x, min_z), (max_x, max_z) = obstacle["bounds"]
    min_x -= float(clearance)
    min_z -= float(clearance)
    max_x += float(clearance)
    max_z += float(clearance)
    return [[min_x, min_z], [max_x, min_z], [max_x, max_z], [min_x, max_z]]


def obstacle_contains_points(x, z, obstacle: Dict[str, Any], clearance: float = 0.0):
    """
    Test whether one or many x/z positions are inside an obstacle footprint.

    Rotated Webots walls are represented as oriented rectangles. Older parsed
    obstacles that only have `bounds` fall back to the legacy AABB behavior.
    """
    x_arr, z_arr = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
    )
    scalar = x_arr.shape == ()

    center = obstacle.get("center")
    axes = obstacle.get("axes")
    half_extents = obstacle.get("half_extents")
    if center is not None and axes is not None and half_extents is not None:
        c = np.asarray(center, dtype=np.float64)
        u = np.asarray(axes[0], dtype=np.float64)
        v = np.asarray(axes[1], dtype=np.float64)
        dx = x_arr - c[0]
        dz = z_arr - c[1]
        local_u = dx * u[0] + dz * u[1]
        local_v = dx * v[0] + dz * v[1]
        inside = (
            (np.abs(local_u) <= float(half_extents[0]) + float(clearance))
            & (np.abs(local_v) <= float(half_extents[1]) + float(clearance))
        )
    elif obstacle.get("corners") is not None and float(clearance) == 0.0:
        inside = _points_in_polygon(x_arr, z_arr, obstacle["corners"])
    else:
        (min_x, min_z), (max_x, max_z) = obstacle["bounds"]
        min_x -= float(clearance)
        min_z -= float(clearance)
        max_x += float(clearance)
        max_z += float(clearance)
        inside = (x_arr >= min_x) & (x_arr <= max_x) & (z_arr >= min_z) & (z_arr <= max_z)

    return bool(inside) if scalar else inside


def obstacle_distance(x: float, z: float, obstacle: Dict[str, Any]) -> float:
    """Return the Euclidean distance from a point to an obstacle footprint."""
    center = obstacle.get("center")
    axes = obstacle.get("axes")
    half_extents = obstacle.get("half_extents")
    if center is not None and axes is not None and half_extents is not None:
        c = np.asarray(center, dtype=np.float64)
        u = np.asarray(axes[0], dtype=np.float64)
        v = np.asarray(axes[1], dtype=np.float64)
        dx = float(x) - c[0]
        dz = float(z) - c[1]
        local_u = dx * u[0] + dz * u[1]
        local_v = dx * v[0] + dz * v[1]
        du = max(abs(local_u) - float(half_extents[0]), 0.0)
        dv = max(abs(local_v) - float(half_extents[1]), 0.0)
        return float(math.hypot(du, dv))

    (min_x, min_z), (max_x, max_z) = obstacle["bounds"]
    dx = max(float(min_x) - float(x), 0.0, float(x) - float(max_x))
    dz = max(float(min_z) - float(z), 0.0, float(z) - float(max_z))
    return 0.0 if dx <= 0.0 and dz <= 0.0 else float(math.hypot(dx, dz))


def _parse_floor_size(world_text: str) -> Tuple[float, float]:
    match = re.search(rf"\bfloorSize\s+({_NUMBER_RE})\s+({_NUMBER_RE})", world_text)
    if not match:
        raise ValueError("Could not find RectangleArena floorSize in WBT file")
    return float(match.group(1)), float(match.group(2))


def _parse_wall_obstacles(world_text: str, world_size: Tuple[float, float]) -> List[Dict[str, Any]]:
    obstacles: List[Dict[str, Any]] = []
    world_half_w = world_size[0] / 2.0
    world_half_h = world_size[1] / 2.0

    for match in re.finditer(r"\bWall\s*\{", world_text):
        block, _ = _extract_braced_block(world_text, match.end() - 1)

        name_match = re.search(r'\bname\s+"([^"]+)"', block)
        name = name_match.group(1) if name_match else None

        if name and name.startswith("OuterWall"):
            continue

        translation_match = re.search(
            rf"\btranslation\s+({_NUMBER_RE})\s+({_NUMBER_RE})\s+({_NUMBER_RE})",
            block,
        )
        size_match = re.search(rf"\bsize\s+({_NUMBER_RE})\s+({_NUMBER_RE})\s+({_NUMBER_RE})", block)
        rotation_match = re.search(
            rf"\brotation\s+({_NUMBER_RE})\s+({_NUMBER_RE})\s+({_NUMBER_RE})\s+({_NUMBER_RE})",
            block,
        )

        if not size_match:
            continue

        if translation_match:
            translation = np.array(
                [
                    float(translation_match.group(1)),
                    float(translation_match.group(2)),
                    float(translation_match.group(3)),
                ],
                dtype=np.float64,
            )
        else:
            translation = np.zeros(3, dtype=np.float64)
        size_x = float(size_match.group(1))
        size_z = float(size_match.group(3))

        if rotation_match:
            axis = (
                float(rotation_match.group(1)),
                float(rotation_match.group(2)),
                float(rotation_match.group(3)),
            )
            angle = float(rotation_match.group(4))
        else:
            axis = (0.0, 1.0, 0.0)
            angle = 0.0

        rotation = _axis_angle_to_matrix(axis, angle)
        half_x = size_x / 2.0
        half_z = size_z / 2.0
        local_corners = np.array(
            [
                [-half_x, 0.0, -half_z],
                [half_x, 0.0, -half_z],
                [half_x, 0.0, half_z],
                [-half_x, 0.0, half_z],
            ],
            dtype=np.float64,
        )
        world_corners = (rotation @ local_corners.T).T + translation
        footprint_corners = world_corners[:, [0, 2]]
        min_x = float(world_corners[:, 0].min())
        max_x = float(world_corners[:, 0].max())
        min_z = float(world_corners[:, 2].min())
        max_z = float(world_corners[:, 2].max())

        axis_x_world = rotation @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis_z_world = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis_x = _normalize_2d(axis_x_world[[0, 2]])
        axis_z = _normalize_2d(axis_z_world[[0, 2]])

        # Ignore boundary walls; the arena size already defines the world extent.
        center_x = (min_x + max_x) / 2.0
        center_z = (min_z + max_z) / 2.0
        spans_width = (max_x - min_x) >= 0.9 * world_size[0]
        spans_height = (max_z - min_z) >= 0.9 * world_size[1]
        near_vertical_edge = abs(abs(center_x) - world_half_w) <= 0.6
        near_horizontal_edge = abs(abs(center_z) - world_half_h) <= 0.6
        if (spans_width and near_horizontal_edge) or (spans_height and near_vertical_edge):
            continue

        obstacle = {
            "type": "rectangle",
            "name": name or f"wall_{len(obstacles)}",
            "bounds": [[min_x, min_z], [max_x, max_z]],
            "corners": footprint_corners.astype(float).tolist(),
            "center": [float(translation[0]), float(translation[2])],
            "half_extents": [float(half_x), float(half_z)],
            "axes": [axis_x, axis_z] if axis_x is not None and axis_z is not None else None,
        }
        obstacles.append(obstacle)

    return obstacles


def _classify_goal_color(color: Tuple[float, float, float]) -> Optional[str]:
    tolerance = 0.15
    for name, target in _GOAL_COLOR_MAP.items():
        if max(abs(component - reference) for component, reference in zip(color, target)) <= tolerance:
            return name
    return None


def _make_unique_name(base_name: str, counts: Dict[str, int]) -> str:
    counts[base_name] = counts.get(base_name, 0) + 1
    if counts[base_name] == 1:
        return base_name
    return f"{base_name}_{counts[base_name]}"


def _parse_goal_markers(world_text: str) -> List[Dict[str, Any]]:
    goals: List[Dict[str, Any]] = []
    goal_name_counts: Dict[str, int] = {}

    for match in re.finditer(r"\bPose\s*\{", world_text):
        block, _ = _extract_braced_block(world_text, match.end() - 1)

        translation_match = re.search(
            rf"\btranslation\s+({_NUMBER_RE})\s+({_NUMBER_RE})\s+({_NUMBER_RE})",
            block,
        )
        cylinder_match = re.search(r"\bgeometry\s+Cylinder\s*\{", block)
        color_match = re.search(
            rf"\bdiffuseColor\s+({_NUMBER_RE})\s+({_NUMBER_RE})\s+({_NUMBER_RE})",
            block,
        )

        if not translation_match or not cylinder_match or not color_match:
            continue

        cylinder_block, _ = _extract_braced_block(block, cylinder_match.end() - 1)
        height_match = re.search(rf"\bheight\s+({_NUMBER_RE})", cylinder_block)
        radius_match = re.search(rf"\bradius\s+({_NUMBER_RE})", cylinder_block)

        translation_y = float(translation_match.group(2))
        cylinder_height = float(height_match.group(1)) if height_match else 0.0

        # Goal markers in these worlds are flat floor cylinders. Ignore raised objects.
        if abs(translation_y) > 0.2 or cylinder_height > 0.05:
            continue

        location = [
            float(translation_match.group(1)),
            float(translation_match.group(3)),
        ]
        color = (
            float(color_match.group(1)),
            float(color_match.group(2)),
            float(color_match.group(3)),
        )
        base_name = _classify_goal_color(color) or "goal"
        goal_name = _make_unique_name(base_name, goal_name_counts)
        radius = float(radius_match.group(1)) if radius_match else _DEFAULT_GOAL_RADIUS

        goals.append(
            {
                "name": goal_name,
                "location": location,
                "radius": radius,
                "color": list(color),
            }
        )

    return goals


def _parse_agent_start(world_text: str) -> Optional[List[float]]:
    match = re.search(r"\bDEF\s+agent\s+Create\s*\{", world_text)
    if not match:
        return None

    block, _ = _extract_braced_block(world_text, match.end() - 1)
    translation_match = re.search(
        rf"\btranslation\s+({_NUMBER_RE})\s+({_NUMBER_RE})\s+({_NUMBER_RE})",
        block,
    )
    if not translation_match:
        return None

    return [
        float(translation_match.group(1)),
        float(translation_match.group(3)),
    ]


def _parse_world_file(world_path: Path) -> Dict[str, Any]:
    world_text = world_path.read_text(encoding="utf-8", errors="ignore")
    size = _parse_floor_size(world_text)
    obstacles = _parse_wall_obstacles(world_text, size)
    goals = _parse_goal_markers(world_text)
    agent_start = _parse_agent_start(world_text)
    return {
        "size": [float(size[0]), float(size[1])],
        "obstacles": obstacles,
        "goals": goals,
        "agent_start": agent_start,
    }


def get_world_config(world_name: str) -> dict:
    """
    Load world configuration directly from a Webots `.wbt` file.

    Args:
        world_name: World file name, stem, or relative path inside `webots/worlds`.

    Returns:
        Dictionary with `size`, `obstacles`, `goals`, and `agent_start` entries.
    """
    resolved_path = _resolve_world_path(world_name)
    cache_key = resolved_path.as_posix()
    if cache_key not in WORLD_CONFIGS:
        WORLD_CONFIGS[cache_key] = _parse_world_file(resolved_path)

    config = copy.deepcopy(WORLD_CONFIGS[cache_key])
    WORLD_CONFIGS.setdefault(Path(world_name).stem, config)
    return config


def get_world_size(world_name: str) -> list:
    """Return the [width, height] of a world."""
    return get_world_config(world_name)["size"]


def get_world_obstacles(world_name: str) -> list:
    """Return the parsed obstacle list for a world."""
    return get_world_config(world_name)["obstacles"]


def get_world_goals(world_name: str) -> list:
    """Return parsed goal markers for a world."""
    return get_world_config(world_name).get("goals", [])


def get_world_agent_start(world_name: str) -> Optional[list]:
    """Return the parsed `[x, z]` start location of the agent, if present."""
    return get_world_config(world_name).get("agent_start")


def build_goal_config_from_world(world_name: str) -> Optional[dict]:
    """Build a controller/driver goal_config directly from parsed world goals."""
    goals = copy.deepcopy(get_world_goals(world_name))
    if not goals:
        return None

    if len(goals) == 1:
        goal = goals[0]
        return {
            "type": "single",
            "name": goal["name"],
            "location": goal["location"],
            "radius": goal["radius"],
        }

    return {
        "type": "multi",
        "goals": [
            {
                "name": goal["name"],
                "location": goal["location"],
                "radius": goal["radius"],
            }
            for goal in goals
        ],
    }
