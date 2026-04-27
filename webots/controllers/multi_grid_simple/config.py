import json
import os

try:
    from robot.robot_mode import RobotMode
except ImportError:  # Support package import from repo root.
    from .robot.robot_mode import RobotMode

MODES_MAP = {
    "LEARN_OJAS": RobotMode.LEARN_OJAS,
    "LEARN_HEBB": RobotMode.LEARN_HEBB,
    "DMTP": RobotMode.DMTP,
    "EXPLOIT": RobotMode.EXPLOIT,
    "PLOTTING": RobotMode.PLOTTING,
}

SELECTED_MODE = "LEARN_OJAS"

MODE_PARAMS = {
    "LEARN_OJAS": {
        "run_time_hours": 5,
        "randomize_start_loc": False,
        "start_loc": [4, -4],
        "goal_location": [-3, 3],
        "max_dist": None,
        "show_bvc_activation": False,
        "enable_ojas": True,
        "enable_stdp": False,
    },
    "LEARN_HEBB": {
        "run_time_hours": 5,
        "randomize_start_loc": False,
        "start_loc": [4, -4],
        "goal_location": [-3, 3],
        "max_dist": None,
        "show_bvc_activation": False,
        "enable_ojas": False,
        "enable_stdp": True,
    },
    "DMTP": {
        "run_time_hours": 5,
        "randomize_start_loc": False,
        "start_loc": [4, -4],
        "goal_location": [-3, 3],
        "max_dist": None,
        "show_bvc_activation": False,
        "enable_ojas": True,
        "enable_stdp": True,
    },
    "EXPLOIT": {
        "run_time_hours": 5,
        "randomize_start_loc": False,
        "start_loc": [4, -4],
        "goal_location": [-3, 3],
        "max_dist": None,
        "show_bvc_activation": False,
        "enable_ojas": False,
        "enable_stdp": False,
    },
    "PLOTTING": {
        "run_time_hours": 1,
        "randomize_start_loc": False,
        "start_loc": [4, -4],
        "goal_location": [-3, 3],
        "max_dist": None,
        "show_bvc_activation": False,
        "enable_ojas": False,
        "enable_stdp": False,
    },
}

EXECUTION_CONFIG = {
    "type": "single",
    "num_trials": 1,
    "reload_world_between_trials": False,
    "pause_on_completion": False,
    "run_id_prefix": None,
}

AUTOMATION_CONFIG = {
    "quit_on_completion": True,
    "pause_on_completion": False,
    "export_image_on_completion": False,
    "completion_image_path": None,
}

LAUNCH_CONFIG = {
    "webots_executable": None,
    "world": "10x10_open.wbt",
    "mode": "fast",
    "batch": True,
    "no_rendering": True,
    "stdout": True,
    "stderr": True,
    "minimize": False,
    "port": None,
    "heartbeat": None,
}

SMOKE_TEST_CONFIG = {
    "world": "10x10_open.wbt",
    "modes": ["PLOTTING", "LEARN_OJAS"],
    "validation_modes": ["LEARN_HEBB", "DMTP", "EXPLOIT"],
    "run_time_hours": 0.001,
    "start_loc": [0, 0],
    "goal_location": [-3, 3],
    "max_dist": 10,
    "series_num_trials": 2,
    "timeout_seconds": 120,
}


def _parse_json_env(var_name: str) -> dict:
    raw = os.environ.get(var_name)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in environment variable {var_name}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Environment variable {var_name} must decode to a JSON object.")
    return parsed


def get_selected_mode() -> str:
    return os.environ.get("MULTI_GRID_SIMPLE_SELECTED_MODE", SELECTED_MODE)


def get_mode_params(mode_key: str) -> dict:
    if mode_key not in MODE_PARAMS:
        raise ValueError(f"No parameters configured for mode '{mode_key}'")
    params = dict(MODE_PARAMS[mode_key])
    params.update(_parse_json_env("MULTI_GRID_SIMPLE_MODE_PARAMS_JSON"))
    return params


def get_execution_config() -> dict:
    config = dict(EXECUTION_CONFIG)
    config.update(_parse_json_env("MULTI_GRID_SIMPLE_EXECUTION_CONFIG_JSON"))
    return config


def get_automation_config() -> dict:
    config = dict(AUTOMATION_CONFIG)
    config.update(_parse_json_env("MULTI_GRID_SIMPLE_AUTOMATION_CONFIG_JSON"))
    return config


def get_launch_config() -> dict:
    config = dict(LAUNCH_CONFIG)
    config.update(_parse_json_env("MULTI_GRID_SIMPLE_LAUNCH_CONFIG_JSON"))
    return config


def get_smoke_test_config() -> dict:
    config = dict(SMOKE_TEST_CONFIG)
    config.update(_parse_json_env("MULTI_GRID_SIMPLE_SMOKE_TEST_CONFIG_JSON"))
    return config
