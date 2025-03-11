# %% Imports and setup
import os
import numpy as np
import pandas as pd
from visualizations.vis_utils import *
from webots.controllers.create3_3D_bvc.visualizations_3D_bvc.hexbins import *


def parse_path_for_model_and_env(full_path: str):
    """
    Extract the model name and environment name from the directory path.
    Extracts model number (e.g., 'model4') and maps rotated_X to envX
    """
    # Extract model number (e.g., model4)
    if "_model" in full_path:
        model_name = full_path.split("_model")[-1]
        model_name = f"model{model_name}"
    else:
        model_name = "unknown_model"

    # Map rotated_X to envX
    if "rotated_1" in full_path:
        env_name = "env1"
    elif "rotated_2" in full_path:
        env_name = "env2"
    elif "rotated_3" in full_path:
        env_name = "env3"
    elif "rotated_4" in full_path:
        env_name = "env4"
    else:
        env_name = "unknown_env"

    return model_name, env_name


def process_hmap(path):
    """
    Process a single hmap directory to calculate cosine similarity sums.
    """
    try:
        print(f"Processing: {path}")
        model_name, env_name = parse_path_for_model_and_env(path)
        print(f"model name is {model_name} and env name is {env_name}")
        # Load data
        hmap_loc, hmap_pcn = load_hmaps_from_dir(
            hmap_names=["hmap_loc", "hmap_pcn"], base_dir=path
        )
        hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

        if hmap_pcn is None or not isinstance(hmap_pcn, np.ndarray):
            raise ValueError(f"hmap_pcn not loaded or invalid for {path}")

        num_cells = hmap_pcn.shape[1]
        all_binned_data_for_env = []

        # Process all cells
        # for cell_index in range(num_cells):
        #    _, _, _, binned_data = create_hexbin(
        #        cell_index=cell_index,
        #        hmap_x=hmap_x,
        #        hmap_y=hmap_y,
        #        hmap_pcn=hmap_pcn,
        #        normalize=True,
        #        filter_bottom_ratio=0.1,
        #        analyze=True,
        #        close_plot=True,
        #    )
        #    all_binned_data_for_env.append(binned_data)

        # Stack binned data
        # stacked_dict = stack_binned_data_by_location(all_binned_data_for_env)

        # Compute cosine similarity sums
        similarity_sums = analyze_cosine_similarity_torch(
            hmap_x=hmap_x, hmap_y=hmap_y, hmap_pcn=hmap_pcn, distance_threshold=2.0
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

    CONTROLLER_NAME = "create3_3D_bvc_looping"

    # Parameters
    root_path = os.path.join(CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl")
    trials = os.listdir(root_path)

    worlds = ["rotated_1", "rotated_2", "rotated_3", "rotated_4"]

    world_trial_paths = {}
    for world in worlds:
        world_trials = [os.path.join(root_path, x) for x in trials if world in x]
        world_trial_paths[world] = world_trials
        print(f"found {len(world_trials)} trials for {world}")
        print(world_trials)

    # Flatten the list of trial paths
    filtered_dirs = []
    for world_trials in world_trial_paths.values():
        filtered_dirs.extend(world_trials)

    print(f"Total trials to process: {len(filtered_dirs)}")

    # %%
    # Process all hmaps sequentially
    results = []
    total_dirs = len(filtered_dirs)

    for i, path in enumerate(filtered_dirs):
        print(f"Processing {i+1}/{total_dirs}: {path}")
        result = process_hmap(path)
        if result is not None:
            results.append(result)

    print(f"Processed {len(results)} hmaps sequentially")

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
                    "model": model_name,  # Will now be like "model4"
                    "environment": env_name,  # Will now be like "env1"
                    "x": x,
                    "y": y,
                    "cosine_similarity_sum": value,
                }
            )

    # Save results to CSV
    df = pd.DataFrame(data_rows)

    output_dir = "webots/controllers/create3_3D_bvc/visualizations_3D_bvc/outputs/"
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
