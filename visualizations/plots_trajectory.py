import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageOps
from typing import Dict, Optional
import json

# Set base directory for all strategies
base_dir = Path(r"C:\Users\bduke\workdir\place_cell_model\AlabiHippocampalModel\analysis\stats")

def plot_preferred_scale_markers(
    background_path: str,
    trial_ids: Dict[str, int],  # Map of strategy name to trial ID
    corner_x: int,
    corner_y: int,
    world_name: str,
    default_trial_id: int = 1,   # Default trial ID if strategy not in trial_ids
    flip_background: bool = True,
    extent=(-10, 10, -10, 10),
    output_dir: str = "analysis/stats",
    step_interval: int = 10
):
    """
    Plots the agent's trajectory:
    - `small_medium_large` strategy: dots with size corresponding to the preferred scale
    - Other strategies: solid black trajectory line

    Parameters:
        background_path: Path to the background image
        trial_ids: Dictionary mapping strategy names to trial IDs
                  e.g. {"small": 1, "medium": 2, "large": 1, "small_medium_large": 3}
        default_trial_id: Trial ID to use for strategies not specified in trial_ids
        ...other parameters remain the same...
    """

    def safe_load_pickle(pkl_path: Path):
        if not pkl_path.exists():
            return None
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    def load_trial_data(strategy_dir: Path, trial_id: int, corner_str: str):
        """Load trial data and check for success"""
        # First load the JSON file to check success
        json_dir = strategy_dir / "JSON"
        json_file = json_dir / f"trial_{trial_id}_corner_{corner_str}.json"
        
        if not json_file.exists():
            print(f"No JSON file found for trial {trial_id} in {strategy_dir.name}")
            return None, None
            
        with open(json_file, 'r') as f:
            trial_data = json.load(f)
            
        if not trial_data.get("success", False):
            print(f"Trial {trial_id} in {strategy_dir.name} was not successful")
            return None, None
            
        # If successful, load the trajectory data
        loc_file = strategy_dir / "hmaps" / f"trial_{trial_id}_corner_{corner_str}_hmap_loc.pkl"
        scale_priority_file = strategy_dir / "hmaps" / f"trial_{trial_id}_corner_{corner_str}_hmap_scale_priority.pkl"
        
        return safe_load_pickle(loc_file), safe_load_pickle(scale_priority_file)

    # Find all available strategy directories - now including world name in path
    base_dir_with_world = base_dir / world_name
    strategy_dirs = [d for d in base_dir_with_world.iterdir() if d.is_dir()]

    # Load and optionally flip the background image
    img = Image.open(background_path)
    if flip_background:
        img = ImageOps.flip(img)
    img = np.array(img)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create one subplot per strategy
    fig, axes = plt.subplots(
        1, len(strategy_dirs),
        figsize=(6 * len(strategy_dirs), 6),
        constrained_layout=True
    )
    if len(strategy_dirs) == 1:
        axes = [axes]

    corner_str = f"{corner_x}_{corner_y}"

    # Define dot sizes for each scale (used only for `small_medium_large`)
    scale_to_size = {0: 3, 1: 6, 2: 9}  # Scale 2 has largest dots

    # Loop over each strategy directory
    for ax, strategy_dir in zip(axes, strategy_dirs):
        strategy_name = strategy_dir.name
        # Get trial ID for this strategy, use default if not specified
        trial_id = trial_ids.get(strategy_name, default_trial_id)
        
        # Load data with success check
        hmap_loc, hmap_scale_priority = load_trial_data(strategy_dir, trial_id, corner_str)

        if hmap_loc is None:
            ax.set_title(f"{strategy_name}\n(No Successful Data\nfor Trial {trial_id})", fontsize=12)
            ax.imshow(img, extent=extent, aspect="auto")
            continue

        # Ensure hmap_scale_priority is 1D (only needed for `small_medium_large`)
        if hmap_scale_priority is not None:
            hmap_scale_priority = np.array(hmap_scale_priority).flatten()

        if hmap_loc.shape[1] == 3:
            hmap_x, hmap_y = hmap_loc[:, 0], hmap_loc[:, 2]
        else:
            hmap_x, hmap_y = hmap_loc[:, 0], hmap_loc[:, 1]

        ax.imshow(img, extent=extent, aspect="auto")
        ax.set_title(f"{strategy_name}\nTrial {trial_id}", fontsize=14)
        ax.set_aspect("equal")

        if strategy_name == "small_medium_large":
            # --- Plot dots for `small_medium_large` ---
            num_steps = min(len(hmap_x), len(hmap_scale_priority))  # Avoid out-of-bounds errors
            for i in range(0, num_steps, step_interval):
                scale_id = int(hmap_scale_priority[i])
                dot_size = scale_to_size.get(scale_id, 3)  # Default to smallest dot if unexpected value
                ax.plot(hmap_x[i], hmap_y[i], 'o', markersize=dot_size, color='black')
        else:
            # --- Plot solid black trajectory line for other strategies ---
            ax.plot(hmap_x, hmap_y, color='black', linewidth=3, linestyle='-')

        # Mark the start with a bigger black dot
        ax.plot(hmap_x[0], hmap_y[0], 'ko', markersize=10, label='Start')

    # Save and show
    # Include trial IDs in filename
    trial_id_str = '_'.join(f"{k}_{v}" for k, v in sorted(trial_ids.items()))
    if not trial_id_str:
        trial_id_str = f"default_{default_trial_id}"
    fig_path = os.path.join(output_dir, f"{world_name}_strategies_trials_{trial_id_str}_corner_{corner_str}.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Saved plot to {fig_path}")
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    # Example usage with different trial IDs for each strategy:
    plot_preferred_scale_markers(
        background_path="images/20x20_2obstacles.png",
        trial_ids={
            "small": 1,
            "medium": 2,
            "large": 1,
            "small_medium_large": 3
        },
        default_trial_id=1,  # Used for any strategy not specified in trial_ids
        corner_x=8,
        corner_y=-8,
        flip_background=True,
        extent=(-10, 10, -10, 10),
        output_dir=r"C:\Users\bduke\workdir\place_cell_model\AlabiHippocampalModel\analysis\plots",
        world_name="20x20_2obstacles",
        step_interval=10
    ) 