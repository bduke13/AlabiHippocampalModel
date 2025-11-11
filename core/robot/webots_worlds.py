"""
Webots World Configuration Definitions

This module contains environment definitions for all Webots worlds including:
- Environment size specifications
- Obstacle definitions (walls, mazes, etc.)
- World bounds and spatial constraints

These configurations are used by:
- Path planning modules for navigation
- Controllers for environment-specific parameters
- Analysis tools for world-specific metrics

Author: Claude Code
"""

# World Configuration Definitions
WORLD_CONFIGS = {
    "20x20_multi_goal": {
        "size": [20.0, 20.0],  # [width, height] in meters
        "obstacles": []  # Open environment - no obstacles
    },

    "20x20_cross_multi_goal": {
        "size": [20.0, 20.0],
        "obstacles": [
            {
                "type": "rectangle",
                "name": "horizontal_wall",
                # Original wall: [-6, 0] to [6, 0] with thickness 0.3
                # Bounds: [-6, -0.15] to [6, 0.15]
                "bounds": [[-6.0, -0.15], [6.0, 0.15]]
            },
            {
                "type": "rectangle",
                "name": "vertical_wall",
                # Original wall: [0, -6] to [0, 6] with thickness 0.3
                # Bounds: [-0.15, -6] to [0.15, 6]
                "bounds": [[-0.15, -6.0], [0.15, 6.0]]
            }
        ]
    },

    "20x20_maze_multi_goal": {
        "size": [20.0, 20.0],
        "obstacles": [
            # Internal maze walls extracted from 20x20_maze_multi_goal.wbt
            {
                "type": "rectangle",
                "name": "MazeMid_HorizLeft",
                # Wall at translation -5, z=3, size 10x0.3 (horizontal)
                # Bounds: x=[-10, 0], z=[2.85, 3.15]
                "bounds": [[-10.0, 2.75], [0.0, 3.25]]
            },
            {
                "type": "rectangle",
                "name": "MazeMid_HorizRight",
                # Wall at translation 5, z=-3, size 10x0.3 (horizontal)
                # Bounds: x=[0, 10], z=[-3.15, -2.85]
                "bounds": [[0.0, -3.25], [10.0, -2.75]]
            },
            {
                "type": "rectangle",
                "name": "MazeMid_VertLeft",
                # Wall at translation -4, z=-0.5, size 7x0.3 (vertical, rotated 90°)
                # Bounds: x=[-4.15, -3.85], z=[-4.0, 3.0]
                "bounds": [[-4.25, -4.0], [-3.75, 3.0]]
            },
            {
                "type": "rectangle",
                "name": "MazeMid_VertRight",
                # Wall at translation 3, z=0.5, size 7x0.3 (vertical, rotated 90°)
                # Bounds: x=[2.85, 3.15], z=[-3.0, 4.0]
                "bounds": [[2.75, -3.0], [3.25, 4.0]]
            },
            {
                "type": "rectangle",
                "name": "MazeBottom_VertCenter",
                # Wall at translation 0, z=-8.5, size 3x0.3 (vertical, rotated 90°)
                # Bounds: x=[-0.15, 0.15], z=[-10.0, -7.0]
                "bounds": [[-0.25, -10.0], [0.25, -7.0]]
            },
            {
                "type": "rectangle",
                "name": "MazeTop_VertCenter",
                # Wall at translation -0.5, z=8.5, size 3x0.3 (vertical, rotated 90°)
                # Bounds: x=[-0.65, -0.35], z=[7.0, 10.0]
                "bounds": [[-0.75, 7.0], [-0.25, 10.0]]
            }
        ]
    },

    "20x20_maze2_multi_goal": {
        "size": [20.0, 20.0],
        "obstacles": [
            # Internal maze walls extracted from 20x20_maze2_multi_goal.wbt
            {
                "type": "rectangle",
                "name": "MazeMid_HorizLeft",
                # Wall at translation -5, z=3, size 10x0.3 (horizontal)
                # Bounds: x=[-10, 0], z=[2.85, 3.15]
                "bounds": [[-10.0, 2.85], [0.0, 3.15]]
            },
            {
                "type": "rectangle",
                "name": "MazeMid_HorizRight",
                # Wall at translation 5, z=-3, size 10x0.3 (horizontal)
                # Bounds: x=[0, 10], z=[-3.15, -2.85]
                "bounds": [[0.0, -3.15], [10.0, -2.85]]
            },
            {
                "type": "rectangle",
                "name": "MazeMid_VertLeft",
                # Wall at translation -4, z=-0.5, size 7x0.3 (vertical, rotated 90°)
                # Bounds: x=[-4.15, -3.85], z=[-4.0, 3.0]
                "bounds": [[-4.15, -4.0], [-3.85, 3.0]]
            },
            {
                "type": "rectangle",
                "name": "MazeMid_VertRight",
                # Wall at translation 3, z=0.5, size 7x0.3 (vertical, rotated 90°)
                # Bounds: x=[2.85, 3.15], z=[-3.0, 4.0]
                "bounds": [[2.85, -3.0], [3.15, 4.0]]
            },
            {
                "type": "rectangle",
                "name": "MazeBottom_VertCenter",
                # Wall at translation 0, z=-8.5, size 3x0.3 (vertical, rotated 90°)
                # Bounds: x=[-0.15, 0.15], z=[-10.0, -7.0]
                "bounds": [[-0.15, -10.0], [0.15, -7.0]]
            },
            {
                "type": "rectangle",
                "name": "MazeTop_VertCenter",
                # Wall at translation -0.5, z=8.5, size 3x0.3 (vertical, rotated 90°)
                # Bounds: x=[-0.65, -0.35], z=[7.0, 10.0]
                "bounds": [[-0.65, 7.0], [-0.35, 10.0]]
            }
        ]
    }
}


def get_world_config(world_name: str) -> dict:
    """
    Get configuration for a specific world.

    Args:
        world_name: Name of the world configuration to retrieve

    Returns:
        Dictionary containing world configuration

    Raises:
        ValueError: If world_name is not found in WORLD_CONFIGS
    """
    if world_name not in WORLD_CONFIGS:
        available = list(WORLD_CONFIGS.keys())
        raise ValueError(f"Unknown world: {world_name}. Available worlds: {available}")

    return WORLD_CONFIGS[world_name].copy()


def list_available_worlds() -> list:
    """
    Get list of all available world configurations.

    Returns:
        List of world names
    """
    return list(WORLD_CONFIGS.keys())


def get_world_size(world_name: str) -> list:
    """
    Get size of a specific world.

    Args:
        world_name: Name of the world

    Returns:
        [width, height] in meters
    """
    config = get_world_config(world_name)
    return config["size"]


def get_world_obstacles(world_name: str) -> list:
    """
    Get obstacles for a specific world.

    Args:
        world_name: Name of the world

    Returns:
        List of obstacle definitions
    """
    config = get_world_config(world_name)
    return config["obstacles"]
