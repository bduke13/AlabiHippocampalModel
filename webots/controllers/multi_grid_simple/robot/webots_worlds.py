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
                # Wall at translation 0, z=-0.25, size 12x0.5 (horizontal)
                # Bounds: x=[-6, 6], z=[-0.25, 0.25]
                "bounds": [[-6.0, -0.25], [6.0, 0.25]]
            },
            {
                "type": "rectangle",
                "name": "vertical_wall",
                # Wall at translation -0.25, z=0, size 12x0.5 (vertical, rotated 90°)
                # Bounds: x=[-0.25, 0.25], z=[-6, 6]
                "bounds": [[-0.25, -6.0], [0.25, 6.0]]
            }
        ]
    },

    "20x20_maze_multi_goal": {
        "size": [20.0, 20.0],
        "obstacles": [
            {
                "type": "rectangle",
                "name": "MazeMid_HorizLeft",
                # Wall at translation -5.25, z=3, rotated 0°, size 9.5x0.5
                # Internal offset: +0.25 in Z → Box center at z=3.25
                # Bounds: x=[-10, -0.5], z=[3.0, 3.5]
                "bounds": [[-10.0, 3.0], [-0.5, 3.5]]
            },
            {
                "type": "rectangle",
                "name": "MazeMid_HorizRight",
                # Wall at translation 5.25, z=-3.2, rotated 0°, size 9.5x0.5
                # Internal offset: +0.25 in Z → Box center at z=-2.95
                # Bounds: x=[0.5, 10], z=[-3.2, -2.7]
                "bounds": [[0.5, -3.2], [10.0, -2.7]]
            },
            {
                "type": "rectangle",
                "name": "MazeMid_VertLeft",
                # Wall at translation -4, z=-0.2, rotated 90°, size 6.4x0.5
                # Internal offset rotates: +0.25 in X → Box center at x=-3.75
                # Bounds: x=[-4.0, -3.5], z=[-3.4, 3.0]
                "bounds": [[-4.0, -3.4], [-3.5, 3.0]]
            },
            {
                "type": "rectangle",
                "name": "MazeMid_VertRight",
                # Wall at translation 3, z=0.5, rotated 90°, size 6.4x0.5
                # Internal offset rotates: +0.25 in X → Box center at x=3.25
                # Bounds: x=[3.0, 3.5], z=[-2.7, 3.7]
                "bounds": [[3.0, -2.7], [3.5, 3.7]]
            },
            {
                "type": "rectangle",
                "name": "MazeBottom_VertCenter",
                # Wall at translation 0.5, z=-8.5, rotated 90°, size 3x0.5
                # Internal offset rotates: +0.25 in X → Box center at x=0.75
                # Bounds: x=[0.5, 1.0], z=[-10, -7]
                "bounds": [[0.5, -10.0], [1.0, -7.0]]
            },
            {
                "type": "rectangle",
                "name": "MazeBottom_VertTop",
                # Wall at translation -1, z=8.5, rotated 90°, size 3x0.5
                # Internal offset rotates: +0.25 in X → Box center at x=-0.75
                # Bounds: x=[-1.0, -0.5], z=[7, 10]
                "bounds": [[-1.0, 7.0], [-0.5, 10.0]]
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
