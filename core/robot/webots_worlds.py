"""World geometry used by the cleaned multiscale controller."""

WORLD_CONFIGS = {
    "20x20": {
        "size": [20.0, 20.0],
        "obstacles": [],
    },
    "20x20_1obstacle": {
        "size": [20.0, 20.0],
        "obstacles": [
            {
                "type": "oriented_rectangle",
                "name": "wall1",
                "center": [0.0, 0.0],
                "size": [7.0, 1.0],
                "yaw": 2.3554,
            },
            {
                "type": "oriented_rectangle",
                "name": "wall2",
                "center": [3.91057, 2.08911],
                "size": [3.0, 1.0],
                "yaw": -2.8797953071795863,
            },
            {
                "type": "oriented_rectangle",
                "name": "wall3",
                "center": [-3.64729, -2.37031],
                "size": [3.0, 1.0],
                "yaw": 0.261799,
            },
        ],
    },
    "20x20_2obstacles": {
        "size": [20.0, 20.0],
        "obstacles": [
            {
                "type": "oriented_rectangle",
                "name": "wall1",
                "center": [4.0, 0.0],
                "size": [7.0, 1.0],
                "yaw": 2.3554,
            },
            {
                "type": "oriented_rectangle",
                "name": "wall2",
                "center": [-4.0, 0.0],
                "size": [7.0, 1.0],
                "yaw": 2.3554,
            },
        ],
    },
    "20x20_goalBehindWall": {
        "size": [20.0, 20.0],
        "obstacles": [
            {
                "type": "oriented_rectangle",
                "name": "wall1",
                "center": [0.0, 0.0],
                "size": [9.0, 1.0],
                "yaw": 2.3554,
            },
        ],
    },
}


def get_world_config(world_name: str) -> dict:
    if world_name not in WORLD_CONFIGS:
        available = ", ".join(sorted(WORLD_CONFIGS))
        raise ValueError(f"Unknown world '{world_name}'. Available worlds: {available}")
    return WORLD_CONFIGS[world_name].copy()


def list_available_worlds() -> list:
    return list(WORLD_CONFIGS.keys())


def get_world_size(world_name: str) -> list:
    return get_world_config(world_name)["size"]


def get_world_obstacles(world_name: str) -> list:
    return get_world_config(world_name)["obstacles"]
