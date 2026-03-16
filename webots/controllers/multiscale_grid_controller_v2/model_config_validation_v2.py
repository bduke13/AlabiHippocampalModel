"""Validation helpers for multiscale_grid_controller_v2 model configuration."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


ALLOWED_RUNTIME_STEP_MODES = {"legacy", "hybrid"}
ALLOWED_NON_EXPLOIT_STOP_CONDITIONS = {"time", "coverage", "time_or_coverage"}
ALLOWED_GOAL_CONSTRAINT_MODES = {"none", "visits", "learning"}
ALLOWED_ADJACENCY_LEARNING_MODES = {"dense", "topk"}
ALLOWED_EXECUTION_TYPES = {"single", "auto"}


@dataclass(frozen=True)
class ScaleConfigV2:
    name: str
    scale_index: int
    num_pc: int
    sigma_r: float
    num_bvc_per_dir: int
    rcn_learning_rate: float
    adjacency_learning_mode: str = "dense"
    adjacency_topk: Optional[int] = None
    adjacency_activity_floor: float = 0.0
    activation_cache_size: int = 0
    activation_cache_quantization: Optional[float] = None


def _require(scale_def: Dict[str, Any], key: str) -> Any:
    if key not in scale_def:
        raise ValueError(f"Scale config missing required key '{key}'")
    return scale_def[key]


def validate_scale_config(scale_def: Dict[str, Any]) -> ScaleConfigV2:
    cfg = ScaleConfigV2(
        name=str(_require(scale_def, "name")),
        scale_index=int(_require(scale_def, "scale_index")),
        num_pc=int(_require(scale_def, "num_pc")),
        sigma_r=float(_require(scale_def, "sigma_r")),
        num_bvc_per_dir=int(_require(scale_def, "num_bvc_per_dir")),
        rcn_learning_rate=float(_require(scale_def, "rcn_learning_rate")),
        adjacency_learning_mode=str(scale_def.get("adjacency_learning_mode", "dense")).lower(),
        adjacency_topk=scale_def.get("adjacency_topk"),
        adjacency_activity_floor=float(scale_def.get("adjacency_activity_floor", 0.0)),
        activation_cache_size=int(scale_def.get("activation_cache_size", 0)),
        activation_cache_quantization=(
            float(scale_def["activation_cache_quantization"])
            if scale_def.get("activation_cache_quantization") is not None
            else None
        ),
    )

    if cfg.num_pc <= 0:
        raise ValueError(f"Scale '{cfg.name}': num_pc must be > 0")
    if cfg.num_bvc_per_dir <= 0:
        raise ValueError(f"Scale '{cfg.name}': num_bvc_per_dir must be > 0")
    if cfg.sigma_r <= 0:
        raise ValueError(f"Scale '{cfg.name}': sigma_r must be > 0")
    if cfg.adjacency_learning_mode not in ALLOWED_ADJACENCY_LEARNING_MODES:
        raise ValueError(
            f"Scale '{cfg.name}': invalid adjacency_learning_mode '{cfg.adjacency_learning_mode}'. "
            f"Allowed: {sorted(ALLOWED_ADJACENCY_LEARNING_MODES)}"
        )
    if cfg.adjacency_topk is not None and int(cfg.adjacency_topk) <= 0:
        raise ValueError(f"Scale '{cfg.name}': adjacency_topk must be > 0 when provided")
    if cfg.adjacency_activity_floor < 0.0:
        raise ValueError(f"Scale '{cfg.name}': adjacency_activity_floor must be >= 0")
    if cfg.activation_cache_size < 0:
        raise ValueError(f"Scale '{cfg.name}': activation_cache_size must be >= 0")
    if cfg.activation_cache_quantization is not None and cfg.activation_cache_quantization <= 0.0:
        raise ValueError(f"Scale '{cfg.name}': activation_cache_quantization must be > 0 when provided")

    return cfg


def validate_scale_configs(scales: List[Dict[str, Any]]) -> None:
    seen = set()
    for scale_def in scales:
        cfg = validate_scale_config(scale_def)
        if cfg.scale_index in seen:
            raise ValueError(f"Duplicate scale_index detected: {cfg.scale_index}")
        seen.add(cfg.scale_index)


def validate_mode_kwargs(kwargs: Dict[str, Any]) -> None:
    runtime_step_mode = str(kwargs.get("runtime_step_mode", "legacy"))
    if runtime_step_mode not in ALLOWED_RUNTIME_STEP_MODES:
        raise ValueError(
            f"Invalid runtime_step_mode '{runtime_step_mode}'. "
            f"Allowed: {sorted(ALLOWED_RUNTIME_STEP_MODES)}"
        )

    stop_cond = str(kwargs.get("non_exploit_stop_condition", "time_or_coverage"))
    if stop_cond not in ALLOWED_NON_EXPLOIT_STOP_CONDITIONS:
        raise ValueError(
            f"Invalid non_exploit_stop_condition '{stop_cond}'. "
            f"Allowed: {sorted(ALLOWED_NON_EXPLOIT_STOP_CONDITIONS)}"
        )

    goal_constraint_mode = str(kwargs.get("goal_constraint_mode", "none"))
    if goal_constraint_mode not in ALLOWED_GOAL_CONSTRAINT_MODES:
        raise ValueError(
            f"Invalid goal_constraint_mode '{goal_constraint_mode}'. "
            f"Allowed: {sorted(ALLOWED_GOAL_CONSTRAINT_MODES)}"
        )

    scale_names = kwargs.get("scale_names", ["small", "medium", "large"])
    if not isinstance(scale_names, list) or len(scale_names) == 0:
        raise ValueError("scale_names must be a non-empty list")

    execution_config = kwargs.get("execution_config")
    if execution_config is not None:
        if not isinstance(execution_config, dict):
            raise ValueError("execution_config must be a dictionary when provided")
        execution_type = str(execution_config.get("type", "single"))
        if execution_type not in ALLOWED_EXECUTION_TYPES:
            raise ValueError(
                f"Invalid execution_config.type '{execution_type}'. "
                f"Allowed: {sorted(ALLOWED_EXECUTION_TYPES)}"
            )
        if execution_type == "auto":
            series_name = execution_config.get("series_name")
            if series_name is not None and not str(series_name).strip():
                raise ValueError("execution_config.series_name must be non-empty when provided")
            num_trials = int(execution_config.get("num_trials", 1))
            if num_trials <= 0:
                raise ValueError("execution_config.num_trials must be > 0")


def validate_mode_param_map(mode_param_map: Dict[str, Dict[str, Any]]) -> None:
    for mode_key, params in mode_param_map.items():
        if not isinstance(params, dict):
            raise ValueError(f"Mode '{mode_key}' parameters must be a dictionary")
        validate_mode_kwargs(params)
