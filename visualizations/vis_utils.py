import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Get the absolute path of the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

CONTROLLER_PATH_PREFIX = str(PROJECT_ROOT / "webots" / "controllers")
CONTROLLER_NAME = "multiscale_grid_controller"
WORLD_NAME = "20x20_1obstacle"
DEFAULT_CONTROLLER_CANDIDATES = ("multiscale_grid_controller_v2", "multiscale_grid_controller")

# Default output directory remains the legacy layout, but helpers below can
# resolve the active controller/world automatically.
OUTPUT_DIR = str(PROJECT_ROOT / "webots" / "controllers" / CONTROLLER_NAME / "pkl" / WORLD_NAME / "vis_outputs")

# Global context for custom paths (can be overridden)
_custom_controller_name = None
_custom_world_name = None

def set_custom_paths(controller_name=None, world_name=None):
    """
    Override the default controller and world names for loading data.

    Args:
        controller_name (str, optional): Custom controller name (e.g., "multiscale_grid_controller")
        world_name (str, optional): Custom world name (e.g., "20x20_maze_multi_goal")
    """
    global _custom_controller_name, _custom_world_name
    _custom_controller_name = controller_name
    _custom_world_name = world_name
    print(f"[vis_utils] Set custom paths: controller={controller_name}, world={world_name}")

def _get_controller_name():
    """Get the active controller name (custom or default)."""
    return _custom_controller_name if _custom_controller_name is not None else CONTROLLER_NAME

def _get_world_name():
    """Get the active world name (custom or default)."""
    return _custom_world_name if _custom_world_name is not None else WORLD_NAME


def _controller_root(controller_name: str) -> Path:
    return PROJECT_ROOT / "webots" / "controllers" / controller_name


def _data_root(controller_name: str, world_name: str) -> Path:
    return _controller_root(controller_name) / "pkl" / world_name


def _discover_world_names(controller_candidates: List[str]) -> List[str]:
    worlds = []
    seen = set()
    for controller_name in controller_candidates:
        pkl_root = _controller_root(controller_name) / "pkl"
        if not pkl_root.exists():
            continue
        for child in sorted(pkl_root.iterdir()):
            if child.is_dir() and child.name not in seen:
                seen.add(child.name)
                worlds.append(child.name)
    return worlds


def resolve_data_context(
    controller_name: Optional[str] = None,
    world_name: Optional[str] = None,
    required_relpaths: Optional[List[str]] = None,
) -> Tuple[str, str, Path]:
    """
    Resolve the controller/world pair that contains the requested data.

    The resolver prefers v2 first, then the legacy controller, but falls back
    to whichever layout actually contains the required files.
    """
    required_relpaths = [Path(p) for p in (required_relpaths or [])]

    controller_candidates: List[str] = []
    if controller_name or _custom_controller_name is not None:
        controller_candidates.append(controller_name or _custom_controller_name)  # type: ignore[arg-type]
    else:
        controller_candidates.extend(DEFAULT_CONTROLLER_CANDIDATES)

    world_candidates: List[str] = []
    if world_name or _custom_world_name is not None:
        world_candidates.append(world_name or _custom_world_name)  # type: ignore[arg-type]
    else:
        world_candidates.append(WORLD_NAME)
        world_candidates.extend(_discover_world_names(controller_candidates))

    # De-duplicate while preserving order.
    controller_candidates = list(dict.fromkeys(controller_candidates))
    world_candidates = list(dict.fromkeys(world_candidates))

    tried = []
    for candidate_controller in controller_candidates:
        for candidate_world in world_candidates:
            data_root = _data_root(candidate_controller, candidate_world)
            tried.append(str(data_root))
            if not data_root.exists():
                continue
            if all((data_root / relpath).exists() for relpath in required_relpaths):
                return candidate_controller, candidate_world, data_root

    raise FileNotFoundError(
        "Could not resolve visualization data context. Tried: " + ", ".join(tried)
    )


def get_output_dir(controller_name: Optional[str] = None, world_name: Optional[str] = None) -> str:
    """Return the vis_outputs directory for the resolved data context."""
    _, _, data_root = resolve_data_context(
        controller_name=controller_name,
        world_name=world_name,
        required_relpaths=["hmaps/hmap_loc.pkl"],
    )
    return str(data_root / "vis_outputs")


def discover_available_scales(controller_name: Optional[str] = None, world_name: Optional[str] = None) -> List[int]:
    """Return the set of available scale indices for the resolved context."""
    _, _, data_root = resolve_data_context(
        controller_name=controller_name,
        world_name=world_name,
        required_relpaths=["hmaps/hmap_loc.pkl"],
    )
    network_dir = data_root / "networks"
    scales = set()
    if network_dir.exists():
        for pkl_path in network_dir.glob("rcn_scale_*.pkl"):
            match = re.match(r"rcn_scale_(\d+)\.pkl$", pkl_path.name)
            if match:
                scales.add(int(match.group(1)))
        if scales:
            return sorted(scales)

        unified_rcn_path = network_dir / "unified_rcn_goal.pkl"
        if unified_rcn_path.exists():
            try:
                with open(unified_rcn_path, "rb") as f:
                    rcn = pickle.load(f)
                scale_configs = list(getattr(rcn, "scale_configs", []) or [])
                if scale_configs:
                    for idx, cfg in enumerate(scale_configs):
                        scales.add(int(cfg.get("scale_index", idx)))
                else:
                    boundaries = list(getattr(rcn, "scale_boundaries", []) or [])
                    if len(boundaries) >= 2:
                        scales.update(range(len(boundaries) - 1))
                if scales:
                    return sorted(scales)
            except Exception:
                pass

    hmap_dir = data_root / "hmaps"
    if hmap_dir.exists():
        for pkl_path in hmap_dir.glob("hmap_pcn_scale_*.pkl"):
            match = re.match(r"hmap_pcn_scale_(\d+)\.pkl$", pkl_path.name)
            if match:
                scales.add(int(match.group(1)))
    return sorted(scales)


def load_layer_pkl(layer_name: str = ""):
    """Loads the layer class object

    Args:
        layer_name: string of the desired layer such as "rcn" or "pcn"

    Returns:
        object: instantiated layer class of specified type

    """
    _, _, data_root = resolve_data_context(
        controller_name=_get_controller_name(),
        world_name=_get_world_name(),
        required_relpaths=[f"networks/{layer_name}.pkl"],
    )
    file_path = data_root / "networks" / f"{layer_name}.pkl"

    with open(file_path, "rb") as f:
        layer = pickle.load(f)
    return layer

# 0: small, l med, 2. lrg
def load_hmaps(
    hmap_names: List[str] = ["hmap_loc", "hmap_pcn_scale_2"],
) -> List[np.ndarray]:
    """
    Load history map (hmap) data from pickle files.

    Args:
        hmap_names: Names of hmaps to load. e.g.: ["hmap_loc", "hmap_pcn"]
    Returns:
        hmaps: arrays containing the heatmap data
        NOTE: if a single hmap name is provided the hmap will not be returned as a list but rather just the np.ndarray for simpler usage
    """
    _, _, data_root = resolve_data_context(
        controller_name=_get_controller_name(),
        world_name=_get_world_name(),
        required_relpaths=[f"hmaps/{hmap}.pkl" for hmap in hmap_names],
    )
    hmap_directory = data_root / "hmaps"

    # collect all hmaps
    hmaps = []
    for hmap in hmap_names:
        hmap_file = f"{hmap}.pkl"
        file_path = hmap_directory / hmap_file
        with open(file_path, "rb") as f:
            # load file
            temp = np.array(pickle.load(f))
            # remove first element from temp
            temp = temp[1:]
            hmaps.append(temp)
            print(f"loaded {hmap} from {file_path}")

    if len(hmap_names) == 1:
        return hmaps[0]

    return hmaps


def convert_xzy_hmaps(hmap_loc: np.ndarray) -> List[np.ndarray]:
    """Utility method that splits hmap_x, hmap_z, and hmap_y from the hmap_loc file output by webots. We use {X,Z,Y} outputs as this is what is returned by webots from the"""
    return hmap_loc[:, 0], hmap_loc[:, 1], hmap_loc[:, 2]


def get_available_directories(root_path: str, max_dirs: int = 200) -> List[str]:
    """
    Recursively find all available directories under the given root path.

    Args:
        root_path (str): The root directory path to start searching from
        max_dirs (int): Maximum number of directories to return before raising an error
                       (default: 200)

    Returns:
        List[str]: List of directory paths relative to the root_path

    Raises:
        ValueError: If the number of directories exceeds max_dirs
    """
    directories = []
    root = Path(root_path)

    if not root.exists():
        raise ValueError(f"Root path {root_path} does not exist")

    root = root.resolve()  # Convert to absolute path
    for path in root.rglob("*"):
        if path.is_dir():
            full_path = str(path) + "/"
            directories.append(full_path)

            if len(directories) > max_dirs:
                raise ValueError(
                    f"Too many directories found (>{max_dirs}). "
                    "Please specify a more specific root path."
                )

    return sorted(directories)


def generate_random_colors(num_cells: int) -> np.ndarray:
    """
    Generate random vibrant colors for visualization.

    Args:
        num_cells (int): Number of colors to generate

    Returns:
        np.ndarray: Array of shape (num_cells, 3) containing RGB colors

    Example:
        >>> colors = generate_random_colors(5)
        >>> print(colors.shape)
        (5, 3)
    """
    # Initialize random colors array
    colors = np.zeros((num_cells, 3))

    for i in range(num_cells):
        while True:
            # Generate random RGB values
            color = np.random.random(3)

            # Ensure at least one channel is very vibrant (>0.8)
            color[np.random.randint(3)] = np.random.uniform(0.8, 1.0)

            # Ensure the color isn't too dark (sum of channels > 1.2)
            if np.sum(color) > 1.2:
                colors[i] = color
                break

    return colors


def filter_directories(directories: List[str], substrings: List[str]) -> List[str]:
    """
    Filter directories to only include those containing any of the given substrings.

    Args:
        directories (List[str]): List of directory paths to filter
        substrings (List[str]): List of substrings to match against directory paths

    Returns:
        List[str]: Filtered list containing only directories that match any of the substrings

    Example:
        >>> dirs = ['/path/trial_1/', '/path/test/', '/path/trial_2/']
        >>> filter_directories(dirs, ['trial'])
        ['/path/trial_1/', '/path/trial_2/']
    """
    filtered_dirs = []
    for directory in directories:
        if any(substring in directory for substring in substrings):
            filtered_dirs.append(directory)
    return sorted(filtered_dirs)

def load_multi_scale_hmaps(scales=None):
    """
    Load history map (hmap) data for multiple scales.
    
    Args:
        scales (List[int], optional): List of scales to load. Defaults to [0, 1, 2].
        
    Returns:
        tuple: (hmap_loc, dict of hmap_pcn by scale)
    """
    if scales is None:
        scales = [0, 1, 2]
    
    # Load location data
    hmap_loc = load_hmaps(["hmap_loc"])
    
    # Load PCN data for each scale
    hmap_pcn_dict = {}
    for scale in scales:
        scale_data = load_hmaps([f"hmap_pcn_scale_{scale}"])
        if isinstance(scale_data, list):
            scale_data = scale_data[0]
        hmap_pcn_dict[scale] = scale_data
    
    return hmap_loc, hmap_pcn_dict

def load_multi_goal_rcn_data(goal_name=None, scale_idx=None):
    """
    Load multi-goal RCN data from the multi_goal_rewards directory.

    Args:
        goal_name (str, optional): Specific goal to load. If None, loads all goals.
        scale_idx (int, optional): Specific scale to load. If None, loads all scales.

    Returns:
        dict: Dictionary with structure {goal_name: {scale_idx: rcn_object}}
              or single rcn_object if both goal_name and scale_idx specified
    """
    _, _, data_root = resolve_data_context(
        controller_name=_get_controller_name(),
        world_name=_get_world_name(),
        required_relpaths=["networks/multi_goal_rewards"],
    )
    multi_goal_dir = data_root / "networks" / "multi_goal_rewards"
    
    if not multi_goal_dir.exists():
        raise FileNotFoundError(f"Multi-goal rewards directory not found: {multi_goal_dir}")
    
    # If both specific goal and scale requested, load just that one
    if goal_name is not None and scale_idx is not None:
        file_path = multi_goal_dir / f"rcn_scale_{scale_idx}_goal_{goal_name}.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"RCN file not found: {file_path}")
        
        with open(file_path, "rb") as f:
            rcn = pickle.load(f)
        print(f"Loaded RCN for {goal_name} scale {scale_idx}")
        return rcn
    
    # Otherwise, load all available combinations
    rcn_data = {}
    
    # Scan directory for RCN files
    for filename in os.listdir(multi_goal_dir):
        if filename.startswith("rcn_scale_") and filename.endswith(".pkl") and "_goal_" in filename:
            # Parse filename: rcn_scale_X_goal_GOALNAME.pkl
            parts = filename.replace(".pkl", "").split("_")
            if len(parts) >= 4:
                try:
                    file_scale_idx = int(parts[2])  # scale_X
                    file_goal_name = "_".join(parts[4:])  # goal_GOALNAME (handle multi-word goal names)
                    
                    # Apply filters if specified
                    if goal_name is not None and file_goal_name != goal_name:
                        continue
                    if scale_idx is not None and file_scale_idx != scale_idx:
                        continue
                    
                    # Load the RCN
                    file_path = multi_goal_dir / filename
                    with open(file_path, "rb") as f:
                        rcn = pickle.load(f)
                    
                    # Store in nested dictionary
                    if file_goal_name not in rcn_data:
                        rcn_data[file_goal_name] = {}
                    rcn_data[file_goal_name][file_scale_idx] = rcn
                    
                    print(f"Loaded RCN for {file_goal_name} scale {file_scale_idx}")
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse filename {filename}: {e}")
                    continue
    
    if not rcn_data:
        raise ValueError("No multi-goal RCN data found matching the specified criteria")
    
    return rcn_data

def load_goal_associations():
    """
    Load goal associations data.

    Returns:
        dict: Goal associations data including place cell mappings
    """
    _, _, data_root = resolve_data_context(
        controller_name=_get_controller_name(),
        world_name=_get_world_name(),
        required_relpaths=["networks/multi_goal_rewards/goal_associations.pkl"],
    )
    multi_goal_dir = data_root / "networks" / "multi_goal_rewards"
    
    associations_path = multi_goal_dir / "goal_associations.pkl"
    
    if not associations_path.exists():
        raise FileNotFoundError(f"Goal associations file not found: {associations_path}")
    
    with open(associations_path, "rb") as f:
        associations = pickle.load(f)
    
    print(f"Loaded goal associations: {list(associations['goal_place_cell_associations'].keys())}")
    return associations

def get_available_multi_goal_combinations():
    """
    Discover available goal-scale combinations in the multi_goal_rewards directory.

    Returns:
        tuple: (goals_list, scales_list, combinations_dict)
               goals_list: List of available goal names
               scales_list: List of available scale indices
               combinations_dict: {goal_name: [scale_indices]}
    """
    _, _, data_root = resolve_data_context(
        controller_name=_get_controller_name(),
        world_name=_get_world_name(),
        required_relpaths=["networks/multi_goal_rewards"],
    )
    multi_goal_dir = data_root / "networks" / "multi_goal_rewards"
    
    if not multi_goal_dir.exists():
        return [], [], {}
    
    goals = set()
    scales = set()
    combinations = {}
    
    # Scan directory for RCN files
    for filename in os.listdir(multi_goal_dir):
        if filename.startswith("rcn_scale_") and filename.endswith(".pkl") and "_goal_" in filename:
            # Parse filename: rcn_scale_X_goal_GOALNAME.pkl
            parts = filename.replace(".pkl", "").split("_")
            if len(parts) >= 4:
                try:
                    scale_idx = int(parts[2])  # scale_X
                    goal_name = "_".join(parts[4:])  # goal_GOALNAME
                    
                    goals.add(goal_name)
                    scales.add(scale_idx)
                    
                    if goal_name not in combinations:
                        combinations[goal_name] = []
                    combinations[goal_name].append(scale_idx)
                    
                except (ValueError, IndexError):
                    continue
    
    # Sort the results
    goals_list = sorted(list(goals))
    scales_list = sorted(list(scales))
    
    # Sort scales for each goal
    for goal in combinations:
        combinations[goal] = sorted(combinations[goal])
    
    print(f"Found {len(goals_list)} goals and {len(scales_list)} scales")
    print(f"Goals: {goals_list}")
    print(f"Scales: {scales_list}")
    
    return goals_list, scales_list, combinations

def load_multi_goal_hmaps(goal_name, scale_idx):
    """
    Load history map data for a specific goal-scale combination.
    
    Args:
        goal_name (str): Name of the goal
        scale_idx (int): Scale index
        
    Returns:
        tuple: (hmap_loc, hmap_pcn) for the specified scale
    """
    # Load location data (shared across all goals/scales)
    hmap_loc = load_hmaps(["hmap_loc"])
    if isinstance(hmap_loc, list):
        hmap_loc = hmap_loc[0]
    
    # Load PCN data for the specific scale
    hmap_pcn = load_hmaps([f"hmap_pcn_scale_{scale_idx}"])
    if isinstance(hmap_pcn, list):
        hmap_pcn = hmap_pcn[0]
    
    return hmap_loc, hmap_pcn
