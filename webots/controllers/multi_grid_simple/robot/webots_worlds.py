"""World metadata for the local multi_grid_simple controller.

These configs intentionally mirror the actual `.wbt` obstacle geometry. Webots
uses the floor plane as X/Z, with Y as height, and the `Wall` proto places its
box geometry offset by half its thickness along local +Z. The helper below
converts Wall translation/size/rotation into world-space rectangle bounds so we
do not hand-maintain mismatched numbers.
"""

from __future__ import annotations


def _wall_bounds(
    *,
    translation_x: float,
    translation_z: float,
    size_x: float,
    size_z: float,
    rotation_y_degrees: int = 0,
) -> list[list[float]]:
    """Convert a Webots Wall proto pose into [min_x, min_z], [max_x, max_z]."""
    if rotation_y_degrees == 0:
        return [
            [translation_x - size_x / 2.0, translation_z],
            [translation_x + size_x / 2.0, translation_z + size_z],
        ]
    if rotation_y_degrees == 90:
        return [
            [translation_x, translation_z - size_x / 2.0],
            [translation_x + size_z, translation_z + size_x / 2.0],
        ]
    if rotation_y_degrees == -90:
        return [
            [translation_x - size_z, translation_z - size_x / 2.0],
            [translation_x, translation_z + size_x / 2.0],
        ]
    if abs(rotation_y_degrees) == 180:
        return [
            [translation_x - size_x / 2.0, translation_z - size_z],
            [translation_x + size_x / 2.0, translation_z],
        ]
    raise ValueError(f"Unsupported wall rotation {rotation_y_degrees}.")


def _rectangle_obstacle(
    name: str,
    *,
    translation_x: float,
    translation_z: float,
    size_x: float,
    size_z: float,
    rotation_y_degrees: int = 0,
) -> dict:
    return {
        "type": "rectangle",
        "name": name,
        "bounds": _wall_bounds(
            translation_x=translation_x,
            translation_z=translation_z,
            size_x=size_x,
            size_z=size_z,
            rotation_y_degrees=rotation_y_degrees,
        ),
        "wall_pose": {
            "translation": [translation_x, translation_z],
            "size": [size_x, size_z],
            "rotation_y_degrees": rotation_y_degrees,
        },
    }


WORLD_CONFIGS = {
    "10x10_open": {
        "size": [10.0, 10.0],
        "world_file": "10x10_open.wbt",
        "obstacles": [],
    },
    "10x10_single_obstacle": {
        "size": [10.0, 10.0],
        "world_file": "10x10_single_obstacle.wbt",
        "obstacles": [
            _rectangle_obstacle(
                "CenterDivider",
                translation_x=-0.25,
                translation_z=0.0,
                size_x=4.0,
                size_z=0.5,
                rotation_y_degrees=90,
            ),
        ],
    },
    "10x10_two_obstacles": {
        "size": [10.0, 10.0],
        "world_file": "10x10_two_obstacles.wbt",
        "obstacles": [
            _rectangle_obstacle(
                "UpperLeftBar",
                translation_x=-2.25,
                translation_z=1.0,
                size_x=4.0,
                size_z=0.5,
                rotation_y_degrees=0,
            ),
            _rectangle_obstacle(
                "LowerRightBar",
                translation_x=1.0,
                translation_z=-1.5,
                size_x=4.0,
                size_z=0.5,
                rotation_y_degrees=90,
            ),
        ],
    },
    "20x20_multi_goal": {
        "size": [20.0, 20.0],
        "world_file": "20x20_multi_goal.wbt",
        "obstacles": [],
    },
    "20x20_cross_multi_goal": {
        "size": [20.0, 20.0],
        "world_file": "20x20_cross_multi_goal.wbt",
        "obstacles": [
            _rectangle_obstacle(
                "horizontal_wall",
                translation_x=0.0,
                translation_z=-0.25,
                size_x=12.0,
                size_z=0.5,
                rotation_y_degrees=0,
            ),
            _rectangle_obstacle(
                "vertical_wall",
                translation_x=-0.25,
                translation_z=0.0,
                size_x=12.0,
                size_z=0.5,
                rotation_y_degrees=90,
            ),
        ],
    },
    "20x20_maze_multi_goal": {
        "size": [20.0, 20.0],
        "world_file": "20x20_maze_multi_goal.wbt",
        "obstacles": [
            _rectangle_obstacle(
                "MazeMid_HorizLeft",
                translation_x=-5.25,
                translation_z=3.0,
                size_x=9.5,
                size_z=0.5,
                rotation_y_degrees=0,
            ),
            _rectangle_obstacle(
                "MazeMid_HorizRight",
                translation_x=5.25,
                translation_z=-3.2,
                size_x=9.5,
                size_z=0.5,
                rotation_y_degrees=0,
            ),
            _rectangle_obstacle(
                "MazeMid_VertLeft",
                translation_x=-4.0,
                translation_z=-0.2,
                size_x=6.4,
                size_z=0.5,
                rotation_y_degrees=90,
            ),
            _rectangle_obstacle(
                "MazeMid_VertRight",
                translation_x=3.0,
                translation_z=0.5,
                size_x=6.4,
                size_z=0.5,
                rotation_y_degrees=90,
            ),
            _rectangle_obstacle(
                "MazeBottom_VertCenter",
                translation_x=0.5,
                translation_z=-8.5,
                size_x=3.0,
                size_z=0.5,
                rotation_y_degrees=90,
            ),
            _rectangle_obstacle(
                "MazeTop_VertCenter",
                translation_x=-1.0,
                translation_z=8.5,
                size_x=3.0,
                size_z=0.5,
                rotation_y_degrees=90,
            ),
        ],
    },
}


def get_world_config(world_name: str) -> dict:
    if world_name not in WORLD_CONFIGS:
        available = list(WORLD_CONFIGS.keys())
        raise ValueError(f"Unknown world: {world_name}. Available worlds: {available}")
    return WORLD_CONFIGS[world_name].copy()


def list_available_worlds() -> list:
    return list(WORLD_CONFIGS.keys())


def get_world_size(world_name: str) -> list:
    return get_world_config(world_name)["size"]


def get_world_obstacles(world_name: str) -> list:
    return get_world_config(world_name)["obstacles"]
