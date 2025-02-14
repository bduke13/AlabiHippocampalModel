# %%
import numpy as np
import pickle
from pathlib import Path
from typing import List
import sys
import os
from pathlib import Path

# Get the absolute path of the project root
# The following two lines helps load pkl files for layers to recognize "core.layers"
project_root = Path(__file__).resolve().parent.parent  # Adjust this if needed
sys.path.append(str(project_root))


CONTROLLER_PATH_PREFIX = "webots/controllers/"
CONTROLLER_NAME = "create3_base"
WORLD_NAME = "10x10"


def load_layer_pkl(layer_name: str = ""):
    """Loads the layer class object

    Args:
        layer_name: string of the desired layer such as "rcn" or "pcn"

    Returns:
        object: instantiated layer class of specified type

    """
    file_path = (
        os.path.join(
            CONTROLLER_PATH_PREFIX,
            CONTROLLER_NAME,
            "pkl",
            WORLD_NAME,
            "networks",
            layer_name,
        )
        + ".pkl"
    )

    with open(file_path, "rb") as f:
        layer = pickle.load(f)
    return layer


def load_hmaps(
    hmap_names: List[str] = ["hmap_loc", "hmap_pcn"],
) -> List[np.ndarray]:
    """
    Load history map (hmap) data from pickle files.

    Args:
        hmap_names: Names of hmaps to load. e.g.: ["hmap_loc", "hmap_pcn"]
    Returns:
        hmaps: arrays containing the heatmap data
        NOTE: if a single hmap name is provided the hmap will not be returned as a list but rather just the np.ndarray for simpler usage
    """
    hmap_directory = os.path.join(
        CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl", WORLD_NAME, "hmaps"
    )

    # collect all hmaps
    hmaps = []
    for hmap in hmap_names:
        hmap_file = f"{hmap}.pkl"
        file_path = os.path.join(hmap_directory, hmap_file)
        with open(file_path, "rb") as f:
            # load file
            temp = np.array(pickle.load(f))
            # remove first element from temp
            temp = temp[1:]
            hmaps.append(temp)

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
