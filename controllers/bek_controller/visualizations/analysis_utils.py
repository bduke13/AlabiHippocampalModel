# %%
import numpy as np
import pickle
import os
import re
from pathlib import Path
from typing import List


def load_hmaps(prefix="three_dim/"):
    """
    Load heatmap data from pickle files.

    Args:
        prefix (str): Directory prefix where the pickle files are located

    Returns:
        tuple: (hmap_x, hmap_y, hmap_z) arrays containing the heatmap data
    """
    with open(f"{prefix}hmap_x.pkl", "rb") as f:
        hmap_x = np.array(pickle.load(f))
    with open(f"{prefix}hmap_y.pkl", "rb") as f:
        hmap_y = np.array(pickle.load(f))
    with open(f"{prefix}hmap_z.pkl", "rb") as f:
        hmap_z = np.array(pickle.load(f))

    return hmap_x, hmap_y, hmap_z


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


def filter_trial_directories(directories: List[str]) -> List[str]:
    """
    Filter directories to only include those with 'trial_X' pattern,
    where X is a number.

    Args:
        directories (List[str]): List of directory paths to filter

    Returns:
        List[str]: Filtered list containing only directories with 'trial_X' pattern
    """
    trial_pattern = re.compile(r".*trial_\d+.*")
    return sorted(
        [directory for directory in directories if trial_pattern.match(directory)]
    )
