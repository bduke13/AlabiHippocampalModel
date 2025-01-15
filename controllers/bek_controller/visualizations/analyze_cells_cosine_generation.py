# gather_hexbin_cosine_data.py

# %% Imports and setup
import os
import numpy as np
import pandas as pd
from collections import defaultdict

# Rely on your project methods (no re-declaration):
from controllers.bek_controller.visualizations.analysis_utils import (
    load_hmaps,
    get_available_directories,
    filter_directories,
)
from controllers.bek_controller.visualizations.hexbins import (
    create_hexbin,
    stack_binned_data_by_location,
    compute_cosine_similarity_sums,
)


def parse_path_for_model_and_env(full_path: str):
    """
    Extract the model name and environment name from the directory path.
    E.g. ".../3D_2L_250_1/inside_shallow/" -> ("3D_2L_250_1", "inside_shallow")
    Adjust this logic if your directory structure differs.
    """
    parts = full_path.strip("/").split("/")
    env_name = parts[-1]  # e.g., "inside_shallow"
    model_name = parts[-2]  # e.g., "3D_2L_250_1"
    return model_name, env_name


# %% Main logic
if __name__ == "__main__":

    # Parameters
    root_path = "controllers/bek_controller/IJCNN"
    desired_path_ends = ["inside_shallow", "inside_medium", "inside_steep", "upright"]

    # 1) Get directories and filter them
    directories = get_available_directories(root_path)
    filtered_dirs = [d for d in directories if "250" in d]
    filtered_dirs = filter_directories(filtered_dirs, desired_path_ends)
    filtered_dirs = [d for d in filtered_dirs if "html_assets" not in d]

    # We'll store data in a dict keyed by (model_name, env_name), or
    # just directly build a list of rows for a DataFrame.
    all_env_data = {}
    global_values = []  # for optional range checking

    # Gather stacked data + compute cosines
    for path in filtered_dirs:
        print(f"Processing: {path}")
        model_name, env_name = parse_path_for_model_and_env(path)

        # Load data from your project
        hmap_x, hmap_y, hmap_z = load_hmaps(path)

        # Combine binned data from ALL cells in this (model, env)
        all_binned_data_for_env = []
        num_cells = hmap_z.shape[1]

        for cell_index in range(num_cells):
            _, _, _, binned_data = create_hexbin(
                cell_index=cell_index,
                hmap_x=hmap_x,
                hmap_y=hmap_y,
                hmap_z=hmap_z,
                normalize=True,
                filter_bottom_ratio=0.1,
                analyze=True,
                close_plot=True,
            )
            all_binned_data_for_env.append(binned_data)

        # Stack them into (x, y) -> [activations across all cells]
        stacked_dict = stack_binned_data_by_location(all_binned_data_for_env)

        # Compute far-distance cosine similarity sums
        similarity_sums = compute_cosine_similarity_sums(
            stacked_dict, distance_threshold=2.0
        )

        # Store in dictionary if you want it in memory:
        all_env_data[(model_name, env_name)] = {
            "stacked_dict": stacked_dict,
            "similarity_sums": similarity_sums,
        }

        # Also extend global range checking
        global_values.extend(list(similarity_sums.values()))

    # Build a rows list for CSV
    data_rows = []
    for (model_name, env_name), env_data in all_env_data.items():
        sim_sums = env_data["similarity_sums"]
        for (x, y), value in sim_sums.items():
            data_rows.append(
                {
                    "model": model_name,
                    "environment": env_name,
                    "x": x,
                    "y": y,
                    "cosine_similarity_sum": value,
                }
            )

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data_rows)

    output_dir = "controllers/bek_controller/analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cosine_similarities.csv")
    df.to_csv(output_path, index=False)

    # Print info
    print(f"\nSaved cosine similarity data to: {output_path}")

    if len(global_values) > 0:
        print(
            f"Global Cosine Range: [min={min(global_values):.3f}, "
            f"max={max(global_values):.3f}]"
        )
    else:
        print("No global_values found (empty dataset?).")
