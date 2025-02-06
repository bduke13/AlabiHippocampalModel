# %%
import numpy as np
import pickle
from pathlib import Path
from typing import List


def load_hmaps(prefix="three_dim/", hmap_names=["hmap_x", "hmap_y", "hmap_pcn"]):
    """
    Load history map (hmap) data from pickle files.

    Args:
        prefix (str): Directory prefix where the pickle files are located

    Returns:
        tuple: (hmap_x, hmap_y, hmap_pcn) arrays containing the heatmap data
    """
    # make sure end directory is a folder

    # collect all hmaps
    hmaps = []
    for hmap in hmap_names:
        with open(f"{prefix}{hmap}.pkl", "rb") as f:
            # load file
            temp = np.array(pickle.load(f))
            # remove first element from temp
            temp = temp[1:]
            hmaps.append(temp)

    return hmaps


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
