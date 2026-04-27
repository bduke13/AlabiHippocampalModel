"""Named runtime parameter profiles for the single-scale controller.

Profiles are intentionally explicit and flat so future tooling can iterate them
for sweeps or grid searches without reverse-engineering driver internals.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Optional


def _arena_diagonal(width: float, height: float) -> float:
    return float((width ** 2 + height ** 2) ** 0.5)


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    arena_label: str
    place_scale_label: str
    description: str
    num_place_cells: int
    num_bvc_per_dir: int
    sigma_r: float
    sigma_theta: float
    n_hd: int = 8
    num_grid_modules: int = 8
    num_grid_cells_per_module: int = 50
    grid_spread_range: tuple[float, float] = (1.2, 1.2)
    grid_scale_multiplier: float = 1.0
    grid_translation_scale: float = 1.0
    grid_threshold: float = 0.7
    grid_mask_resolution: int = 128
    grid_smooth_sigma: float = 1.5
    timestep: int = 32 * 3
    tau_w: int = 5
    max_speed: int = 16
    wheel_radius: float = 0.031
    axle_length: float = 0.271756
    max_dist: float = 15.0
    goal_r_explore: float = 0.3
    goal_r_exploit: float = 0.5


RUNTIME_PROFILES: dict[str, RuntimeProfile] = {
    "10x10_place_small": RuntimeProfile(
        name="10x10_place_small",
        arena_label="10x10",
        place_scale_label="small",
        description="Current baseline single-scale profile for the 10x10 environments.",
        num_place_cells=500,
        num_bvc_per_dir=50,
        sigma_r=0.5,
        sigma_theta=1.0,
        num_grid_modules=8,
        num_grid_cells_per_module=50,
        grid_spread_range=(1.2, 1.2),
        grid_scale_multiplier=1.0,
        grid_translation_scale=1.0,
        grid_threshold=0.7,
        grid_mask_resolution=128,
        grid_smooth_sigma=1.5,
        max_dist=_arena_diagonal(10.0, 10.0),
    ),
    "20x20_place_small": RuntimeProfile(
        name="20x20_place_small",
        arena_label="20x20",
        place_scale_label="small",
        description="Small place fields for the 20x20 environments, derived from the legacy multiscale controller.",
        num_place_cells=2000,
        num_bvc_per_dir=100,
        sigma_r=0.5,
        sigma_theta=1.0,
        num_grid_modules=8,
        num_grid_cells_per_module=100,
        grid_spread_range=(1.5, 1.5),
        grid_scale_multiplier=4.0,
        grid_translation_scale=2.0,
        grid_threshold=0.7,
        grid_mask_resolution=256,
        grid_smooth_sigma=1.0,
        max_dist=_arena_diagonal(20.0, 20.0),
    ),
    "20x20_place_medium": RuntimeProfile(
        name="20x20_place_medium",
        arena_label="20x20",
        place_scale_label="medium",
        description="Medium place fields for the 20x20 environments, derived from the legacy multiscale controller.",
        num_place_cells=1000,
        num_bvc_per_dir=50,
        sigma_r=1.0,
        sigma_theta=3.0,
        num_grid_modules=8,
        num_grid_cells_per_module=50,
        grid_spread_range=(1.5, 1.5),
        grid_scale_multiplier=5.5,
        grid_translation_scale=2.0,
        grid_threshold=0.7,
        grid_mask_resolution=128,
        grid_smooth_sigma=1.5,
        max_dist=_arena_diagonal(20.0, 20.0),
    ),
    "20x20_place_large": RuntimeProfile(
        name="20x20_place_large",
        arena_label="20x20",
        place_scale_label="large",
        description="Large place fields for the 20x20 environments, derived from the legacy multiscale controller.",
        num_place_cells=500,
        num_bvc_per_dir=75,
        sigma_r=1.5,
        sigma_theta=5.0,
        num_grid_modules=8,
        num_grid_cells_per_module=50,
        grid_spread_range=(1.5, 1.5),
        grid_scale_multiplier=7.0,
        grid_translation_scale=2.0,
        grid_threshold=0.7,
        grid_mask_resolution=128,
        grid_smooth_sigma=1.5,
        max_dist=_arena_diagonal(20.0, 20.0),
    ),
}

_AUTO_PROFILE_BY_WORLD_PREFIX: tuple[tuple[str, str], ...] = (
    ("20x20_", "20x20_place_medium"),
    ("10x10_", "10x10_place_small"),
)

_RUNTIME_METADATA_KEYS = {
    "name",
    "arena_label",
    "place_scale_label",
    "description",
}


def list_runtime_profile_names() -> list[str]:
    return list(RUNTIME_PROFILES.keys())


def get_runtime_profile(name: str) -> RuntimeProfile:
    if name not in RUNTIME_PROFILES:
        raise ValueError(
            f"Unknown runtime profile '{name}'. Available profiles: {list_runtime_profile_names()}"
        )
    return RUNTIME_PROFILES[name]


def choose_default_runtime_profile(world_name: Optional[str]) -> str:
    if world_name:
        for prefix, profile_name in _AUTO_PROFILE_BY_WORLD_PREFIX:
            if world_name.startswith(prefix):
                return profile_name
    return "10x10_place_small"


def _profile_to_dict(profile: RuntimeProfile) -> dict[str, Any]:
    return asdict(profile)


def runtime_parameter_keys() -> list[str]:
    sample = _profile_to_dict(next(iter(RUNTIME_PROFILES.values())))
    return sorted(key for key in sample.keys() if key not in _RUNTIME_METADATA_KEYS)


def resolve_runtime_profile(
    *,
    requested_name: Optional[str],
    world_name: Optional[str],
    overrides: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    profile_name = requested_name or choose_default_runtime_profile(world_name)
    resolved = deepcopy(_profile_to_dict(get_runtime_profile(profile_name)))

    if overrides:
        unknown_keys = sorted(set(overrides) - set(resolved))
        if unknown_keys:
            raise ValueError(
                "Unknown runtime profile override keys: "
                f"{unknown_keys}. Valid keys: {sorted(resolved.keys())}"
            )
        resolved.update(deepcopy(overrides))

    return profile_name, resolved
