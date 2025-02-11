# %% Imports and setup
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial

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
    Adjust this logic if your directory structure differs.
    """
    parts = full_path.strip("/").split("/")
    env_name = parts[-1]
    model_name = parts[-2]
    return model_name, env_name


def process_hmap(path):
    """
    Process a single hmap directory to calculate cosine similarity sums.
    """
    try:
        print(f"Processing: {path}")
        model_name, env_name = parse_path_for_model_and_env(path)

        # Load data
        hmap_x, hmap_y, hmap_z = load_hmaps(
            path, hmap_names=["hmap_x", "hmap_y", "hmap_pcn"]
        )
        if hmap_z is None or not isinstance(hmap_z, np.ndarray):
            raise ValueError(f"hmap_z not loaded or invalid for {path}")

        num_cells = hmap_z.shape[1]
        all_binned_data_for_env = []

        # Process all cells
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

        # Stack binned data
        stacked_dict = stack_binned_data_by_location(all_binned_data_for_env)

        # Compute cosine similarity sums
        similarity_sums = compute_cosine_similarity_sums(
            stacked_dict, distance_threshold=2.0
        )

        return {
            "model_name": model_name,
            "env_name": env_name,
            "similarity_sums": similarity_sums,
        }

    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


if __name__ == "__main__":

    # Parameters
    root_path = "controllers/bek_controller/IJCNN"
    desired_path_ends = ["inside_shallow", "inside_medium", "inside_steep", "upright"]

    # Get and filter directories
    directories = get_available_directories(root_path)
    filtered_dirs = [d for d in directories if "250" in d]
    filtered_dirs = filter_directories(filtered_dirs, desired_path_ends)
    filtered_dirs = [d for d in filtered_dirs if "html_assets" not in d]

    # Process all hmaps in parallel
    num_processes = max(1, cpu_count() - 1)  # Leave at least one CPU free
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_hmap, filtered_dirs)

    # Filter out None results
    results = [r for r in results if r is not None]
    print(f"Processed {len(results)} hmaps using {num_processes} processes")

    # Consolidate results
    all_env_data = {}
    global_values = []

    for result in results:
        model_name = result["model_name"]
        env_name = result["env_name"]
        similarity_sums = result["similarity_sums"]

        all_env_data[(model_name, env_name)] = {"similarity_sums": similarity_sums}
        global_values.extend(list(similarity_sums.values()))

    # Build rows for CSV
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

    # Save results to CSV
    df = pd.DataFrame(data_rows)
    output_dir = "controllers/bek_controller/analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cosine_similarities.csv")
    df.to_csv(output_path, index=False)

    print(f"\nSaved cosine similarity data to: {output_path}")

    if global_values:
        print(
            f"Global Cosine Range: [min={min(global_values):.3f}, max={max(global_values):.3f}]"
        )
    else:
        print("No global_values found (empty dataset?).")
