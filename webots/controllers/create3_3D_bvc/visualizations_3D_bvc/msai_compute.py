# %% Imports and setup
import os
import numpy as np
import pandas as pd
from visualizations.vis_utils import *
from webots.controllers.create3_3D_bvc.visualizations_3D_bvc.hexbins import *


def get_world_from_path(full_path: str):
    """
    Extract the world name from the path.
    """
    for world, _, _ in experiments:
        if world in full_path:
            return world
    return "unknown_world"


def parse_path_for_model_and_env(full_path: str):
    """
    Extract the model name and environment name from the directory path
    based on the experiments configuration.
    """
    world_name = get_world_from_path(full_path)

    # Find the matching experiment
    for world, env_name, model_name in experiments:
        if world == world_name:
            return model_name, env_name, world_name

    # Fallback to old method if no match found
    if "_model" in full_path:
        model_name = full_path.split("_model")[-1]
        model_name = f"model{model_name}"
    else:
        model_name = "unknown_model"

    return model_name, "unknown_env", world_name


def process_hmap(path):
    """
    Process a single hmap directory to calculate cosine similarity sums.
    """
    try:
        print(f"Processing: {path}")
        model_name, env_name, world_name = parse_path_for_model_and_env(path)
        print(f"model name: {model_name}, env name: {env_name}, world: {world_name}")
        # Load data
        hmap_loc, hmap_pcn = load_hmaps_from_dir(
            hmap_names=["hmap_loc", "hmap_pcn"], base_dir=path
        )
        hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)

        if hmap_pcn is None or not isinstance(hmap_pcn, np.ndarray):
            raise ValueError(f"hmap_pcn not loaded or invalid for {path}")

        # Compute cosine similarity sums
        similarity_sums = analyze_cosine_similarity_torch(
            hmap_x=hmap_x, hmap_y=hmap_y, hmap_pcn=hmap_pcn, distance_threshold=2.0
        )

        return {
            "model_name": model_name,
            "env_name": env_name,
            "world_name": world_name,
            "similarity_sums": similarity_sums,
        }

    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


if __name__ == "__main__":

    CONTROLLER_NAME = "create3_3D_bvc"

    # Parameters
    root_path = os.path.join(CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl")
    trials = os.listdir(root_path)
    # Define experiments as [world, env_name, model_name]

    experiments_base = [["10x10_open", "env1", "base"], ["10x10_cross", "env2", "base"]]

    experiments_obstacles = [
        ["3D_bvc_two_shapes_control", "env1", "3D_bvc"],
        ["3D_bvc_two_shapes_test", "env2", "3D_bvc"],
    ]

    experiments_ceilings = [
        ["3D_bvc_ceilings_control", "env1", "3D_bvc"],
        ["3D_bvc_ceilings_test", "env2", "3D_bvc"],
    ]

    experiments_scaling = [
        ["3D_bvc_cross_rotated_1_normal", "env1", "normal"],
        ["3D_bvc_cross_rotated_2_no_vert_scaling", "env2", "no_vert"],
        ["3D_bvc_cross_rotated_3_no_dist_scaling", "env3", "no_dist"],
        ["3D_bvc_cross_rotated_4_no_vert_or_dist_scaling", "env4", "no_vert_dist"],
    ]

    experiments_rotated = [
        ["3D_bvc_cross_rotated_1_model0", "env1", "model0"],
        ["3D_bvc_cross_rotated_1_model1", "env1", "model1"],
        ["3D_bvc_cross_rotated_1_model2", "env1", "model2"],
        ["3D_bvc_cross_rotated_1_model3", "env1", "model3"],
        ["3D_bvc_cross_rotated_1_model4", "env1", "model4"],
        ["3D_bvc_cross_rotated_2_model0", "env2", "model0"],
        ["3D_bvc_cross_rotated_2_model1", "env2", "model1"],
        ["3D_bvc_cross_rotated_2_model2", "env2", "model2"],
        ["3D_bvc_cross_rotated_2_model3", "env2", "model3"],
        ["3D_bvc_cross_rotated_2_model4", "env2", "model4"],
        ["3D_bvc_cross_rotated_3_model0", "env3", "model0"],
        ["3D_bvc_cross_rotated_3_model1", "env3", "model1"],
        ["3D_bvc_cross_rotated_3_model2", "env3", "model2"],
        ["3D_bvc_cross_rotated_3_model3", "env3", "model3"],
        ["3D_bvc_cross_rotated_3_model4", "env3", "model4"],
        ["3D_bvc_cross_rotated_4_model0", "env4", "model0"],
        ["3D_bvc_cross_rotated_4_model1", "env4", "model1"],
        ["3D_bvc_cross_rotated_4_model2", "env4", "model2"],
        ["3D_bvc_cross_rotated_4_model3", "env4", "model3"],
        ["3D_bvc_cross_rotated_4_model4", "env4", "model4"],
    ]

    experiments = experiments_obstacles

    # Extract just the world names for filtering
    worlds = [world for world, _, _ in experiments]

    # Print experiment configuration
    print("Experiment Configuration:")
    for world, env_name, model_name in experiments:
        print(f"  {world} -> {env_name} ({model_name})")

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
        world_name = result["world_name"]
        similarity_sums = result["similarity_sums"]

        all_env_data[(model_name, env_name, world_name)] = {
            "similarity_sums": similarity_sums
        }
        global_values.extend(list(similarity_sums.values()))

    data_rows = []
    for (model_name, env_name, world_name), env_data in all_env_data.items():
        sim_sums = env_data["similarity_sums"]
        for (x, y), value in sim_sums.items():
            data_rows.append(
                {
                    "model": model_name,
                    "environment": env_name,
                    "world": world_name,
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

    # Print experiment mapping for reference
    print("\nExperiment mapping:")
    for world, env_name, model_name in experiments:
        print(f"  {env_name} = {model_name} ({world})")

    if global_values:
        print(
            f"Global Cosine Range: [min={min(global_values):.3f}, max={max(global_values):.3f}]"
        )
    else:
        print("No global_values found (empty dataset?).")
