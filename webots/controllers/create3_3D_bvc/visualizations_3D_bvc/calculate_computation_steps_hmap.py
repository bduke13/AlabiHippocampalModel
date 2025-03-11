# %% Imports and setup
import os
import numpy as np
import pandas as pd
import pickle
from visualizations.vis_utils import *
from webots.controllers.create3_3D_bvc.visualizations_3D_bvc.hexbins import *


if __name__ == "__main__":
    CONTROLLER_NAME = "create3_3D_bvc"

    # Parameters
    root_path = os.path.join(CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl")

    # Get all directories in the root path
    all_dirs = [
        os.path.join(root_path, d)
        for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d))
    ]

    # Store results
    results = []

if __name__ == "__main__":
    CONTROLLER_NAME = "create3_3D_bvc"

    # Parameters
    root_path = os.path.join(CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl")

    # Get all directories in the root path
    all_dirs = [
        os.path.join(root_path, d)
        for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d))
    ]

    # Store results
    results = []

    # Process each directory
    for dir_path in all_dirs:
        hmap_loc, hmap_pcn = load_hmaps_from_dir(
            hmap_names=["hmap_loc", "hmap_pcn"], base_dir=all_dirs[0]
        )
        results.append(hmap_pcn.shape[0])
        print(hmap_pcn.shape[0])

    print(np.mean(results))
