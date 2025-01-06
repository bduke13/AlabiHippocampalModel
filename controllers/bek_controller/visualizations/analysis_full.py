# %%
import pandas as pd
import numpy as np
from pathlib import Path
from controllers.bek_controller.visualizations.analysis_utils import (
    load_hmaps,
    get_available_directories,
    filter_trial_directories,
)
from controllers.bek_controller.visualizations.analyze_cells_2 import (
    calculate_far_cosine_metric,
)
from controllers.bek_controller.visualizations.analyze_cells import (
    analyze_far_field_activations,
)
from controllers.bek_controller.visualizations.overlayed_cells import (
    plot_overlayed_cells,
)

# %% Set pandas display options
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", None)  # Width of the display in characters
pd.set_option("display.max_colwidth", None)  # Show full contents of each column

# %% # get all trial paths
directories = get_available_directories(
    root_path="controllers/bek_controller/paper_data/"
)
trial_paths = filter_trial_directories(directories=directories)
# Filter for only trial_0 paths
trial_paths = [path for path in trial_paths if "trial_0" in path]
print(f"{len(trial_paths)} trial_0 paths to process")

# %%
# Initialize lists to store results
results = []

# Process each trial
for path in trial_paths:
    print(f"Processing {path}")

    # Load data
    try:
        hmap_x, hmap_y, hmap_z = load_hmaps(path)

        # Calculate metrics
        far_field_stats = analyze_far_field_activations(
            hmap_x, hmap_y, hmap_z, distance_threshold=2.0
        )
        cosine_stats = calculate_far_cosine_metric(
            hmap_x, hmap_y, hmap_z, distance_threshold=2.0
        )

        # Extract path components
        path_parts = Path(path).parts
        for i, part in enumerate(path_parts):
            if part == "paper_data":
                model_name = path_parts[i + 1]
                world_name = path_parts[i + 2]
                trial_name = path_parts[i + 3]
                break

        # Calculate summary statistics
        total_far_field = sum(stat["total_far_activation"] for stat in far_field_stats)
        avg_far_field = total_far_field / len(far_field_stats)

        cosine_values = np.array(cosine_stats["far_cosine_values"])
        avg_cosine = np.mean(cosine_values) if len(cosine_values) > 0 else 0
        max_cosine = np.max(cosine_values) if len(cosine_values) > 0 else 0

        # Store results
        results.append(
            {
                "model_name": model_name,
                "world_name": world_name,
                "trial_name": trial_name,
                "total_far_field": total_far_field,
                "avg_far_field": avg_far_field,
                "avg_cosine_similarity": avg_cosine,
                "max_cosine_similarity": max_cosine,
                "num_far_pairs": len(cosine_values),
            }
        )
        print(f"processed {path}")

    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        continue

# Create DataFrame
df = pd.DataFrame(results)
print("DONE WITH ANALYSIS")
# %%
# Display results
print("\nAnalysis Results:")
print(df)


# %%
# Save to CSV
df.to_csv("trial_analysis_results.csv", index=False)
print("\nResults saved to trial_analysis_results.csv")


# %%
# Load from CSV
def load_analysis_results():
    """Load the saved analysis results"""
    return pd.read_csv("trial_analysis_results.csv")


# Load the results
df = load_analysis_results()
print("Loaded analysis results from trial_analysis_results.csv")
print(df)

# %%
# PLOTS HERE
import seaborn as sns
import matplotlib.pyplot as plt

# Define the desired order of models
model_order = ["2D", "3D_Base", "3D_1L", "3D_2L", "3D_5L", "3D_10L"]

# Filter for world_4 only and sort according to model order
world_4_df = df[df["world_name"] == "world_2"].copy()
world_4_df["model_name"] = pd.Categorical(
    world_4_df["model_name"], categories=model_order, ordered=True
)
world_4_df = world_4_df.sort_values("model_name")

sns.set_palette("husl")

# Create figure with larger size
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))

# Plot 1: Average Cosine Similarity for World 4
sns.barplot(
    data=world_4_df,
    x="model_name",
    y="avg_cosine_similarity",
    ax=ax1,
    order=model_order,
)
ax1.set_title("Average Cosine Similarity in World 3")
ax1.set_xlabel("Model")
ax1.set_ylabel("Average Cosine Similarity")
ax1.tick_params(axis="x", rotation=45)

# Plot 2: Average Far Field for World 4
sns.barplot(
    data=world_4_df, x="model_name", y="avg_far_field", ax=ax2, order=model_order
)
ax2.set_title("Average Far Field in World 3")
ax2.set_xlabel("Model")
ax2.set_ylabel("Average Far Field")
ax2.tick_params(axis="x", rotation=45)

# Plot 3: Number of Far Pairs
sns.barplot(
    data=world_4_df, x="model_name", y="num_far_pairs", ax=ax3, order=model_order
)
ax3.set_title("Number of Far Pairs in World 3")
ax3.set_xlabel("Model")
ax3.set_ylabel("Number of Far Pairs")
ax3.tick_params(axis="x", rotation=45)

# Adjust layout and display
plt.tight_layout()
plt.show()


# %% Plot Overlayed Cells
print("\nGenerating Overlayed Cell Plots:")
for path in trial_paths:
    print(f"\nProcessing overlayed cells for {path}")
    try:
        # Load data
        hmap_x, hmap_y, hmap_z = load_hmaps(path)

        # Extract path components for labeling
        path_parts = Path(path).parts
        for i, part in enumerate(path_parts):
            if part == "paper_data":
                model_name = path_parts[i + 1]
                world_name = path_parts[i + 2]
                trial_name = path_parts[i + 3]
                break

        # Create suffix for the plot filename
        plot_suffix = f"_{model_name}_{world_name}_{trial_name}"

        # Generate overlayed cells plot
        plot_overlayed_cells(
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            hmap_z=hmap_z,
            colors_path="controllers/bek_controller/visualizations/colors.json",
            gridsize=100,
            save_plot=True,
            suffix=plot_suffix,
        )

    except Exception as e:
        print(f"Error processing overlayed cells for {path}: {str(e)}")
        continue
