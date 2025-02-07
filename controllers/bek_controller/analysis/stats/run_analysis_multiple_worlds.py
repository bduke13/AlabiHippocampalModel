import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the base directories for "vanilla" and "multiscale" modes
vanilla_base_dir = r"controllers/bek_controller/analysis/stats/vanilla"
multiscale_base_dir = r"controllers/bek_controller/analysis/stats/multiscale"

# Define the worlds to analyze
worlds = [
    "world0_20x20",
    "world0_20x20-obstacles",
    "world0_20x20-goalBehindWall",
    "world0_20x20-2obstacles"
]

# Specify the corner to analyze
corner_to_analyze = (8, -8)

# Load JSON files for a specific corner
def load_json_for_corner(directory, corner):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            try:
                parts = filename.split("_")
                x_coord = int(parts[-2])
                y_coord = int(parts[-1].split(".")[0])
                if (x_coord, y_coord) != corner:
                    continue
            except (IndexError, ValueError):
                print(f"Error parsing corner from filename: {filename}")
                continue
            
            with open(os.path.join(directory, filename), 'r') as f:
                file_data = json.load(f)
                data.append(file_data)
    return pd.DataFrame(data)

# Metrics to compare
metrics = ['total_distance_traveled', 'total_time_secs', 'turn_count']

# Create a DataFrame to store data for comparison
comparison_data = {
    "world": [],
    "mode": [],
    **{f"{metric}_mean": [] for metric in metrics},
    **{f"{metric}_sem": [] for metric in metrics},
    "success_rate_mean": [],
    "success_rate_sem": []
}

# Iterate through each world
for world in worlds:
    # Define directories for the current world
    vanilla_dir = os.path.join(vanilla_base_dir, world, "JSON")
    multiscale_dir = os.path.join(multiscale_base_dir, world, "JSON")
    
    # Load data for the specified corner in both modes
    vanilla_df = load_json_for_corner(vanilla_dir, corner_to_analyze)
    multiscale_df = load_json_for_corner(multiscale_dir, corner_to_analyze)
    
    # Process data for vanilla mode
    if not vanilla_df.empty:
        comparison_data["world"].append(world)
        comparison_data["mode"].append("vanilla")
        for metric in metrics:
            comparison_data[f"{metric}_mean"].append(vanilla_df[metric].mean())
            comparison_data[f"{metric}_sem"].append(vanilla_df[metric].sem())
        comparison_data["success_rate_mean"].append(vanilla_df["success"].mean())
        comparison_data["success_rate_sem"].append(vanilla_df["success"].sem())
    else:
        print(f"No data found for vanilla in {world}")
    
    # Process data for multiscale mode
    if not multiscale_df.empty:
        comparison_data["world"].append(world)
        comparison_data["mode"].append("multiscale")
        for metric in metrics:
            comparison_data[f"{metric}_mean"].append(multiscale_df[metric].mean())
            comparison_data[f"{metric}_sem"].append(multiscale_df[metric].sem())
        comparison_data["success_rate_mean"].append(multiscale_df["success"].mean())
        comparison_data["success_rate_sem"].append(multiscale_df["success"].sem())
    else:
        print(f"No data found for multiscale in {world}")

# Convert comparison data to a DataFrame
comparison_df = pd.DataFrame(comparison_data)

# Plot comparison of metrics across worlds for the specified corner
output_dir = "controllers/bek_controller/analysis/stats/comparison"
os.makedirs(output_dir, exist_ok=True)

for metric in metrics:
    plt.figure(figsize=(10, 6))
    
    # Filter data for the current metric
    metric_data = comparison_df[["world", "mode", f"{metric}_mean", f"{metric}_sem"]].copy()
    
    # Separate vanilla and multiscale data
    vanilla_data = metric_data[metric_data["mode"] == "vanilla"]
    multiscale_data = metric_data[metric_data["mode"] == "multiscale"]
    
    # Ensure consistent ordering of worlds
    x = np.arange(len(worlds))  # Indices for worlds
    vanilla_means = [
        vanilla_data[vanilla_data["world"] == world][f"{metric}_mean"].values[0] if world in vanilla_data["world"].values else 0
        for world in worlds
    ]
    vanilla_sems = [
        vanilla_data[vanilla_data["world"] == world][f"{metric}_sem"].values[0] if world in vanilla_data["world"].values else 0
        for world in worlds
    ]
    multiscale_means = [
        multiscale_data[multiscale_data["world"] == world][f"{metric}_mean"].values[0] if world in multiscale_data["world"].values else 0
        for world in worlds
    ]
    multiscale_sems = [
        multiscale_data[multiscale_data["world"] == world][f"{metric}_sem"].values[0] if world in multiscale_data["world"].values else 0
        for world in worlds
    ]

    # Define x positions for the bars
    width = 0.35

    # Plot vanilla bars with error bars
    plt.bar(
        x - width / 2, 
        vanilla_means, 
        width=width, 
        yerr=vanilla_sems, 
        label="Vanilla", 
        alpha=0.8, 
        color="black", 
        capsize=5, 
        edgecolor="gray",
        linewidth=1,
        zorder=3
    )
    # Plot multiscale bars with error bars
    plt.bar(
        x + width / 2, 
        multiscale_means, 
        width=width, 
        yerr=multiscale_sems, 
        label="Multiscale", 
        alpha=0.8, 
        color="#87CEEB",  # Light blue
        capsize=5,
        zorder=3
    )
    
    # Add labels, title, and legend
    plt.xticks(x, worlds, rotation=45)
    plt.ylabel(metric.replace("_", " ").capitalize())
    plt.title(f"Comparison of {metric.replace('_', ' ').capitalize()} for Corner {corner_to_analyze}")
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    
    # Save the plot (commented out)
    # save_path = os.path.join(output_dir, f"comparison_{metric}_corner_{corner_to_analyze[0]}_{corner_to_analyze[1]}.png")
    # plt.savefig(save_path)
    
    # Show the plot
    plt.show()
    plt.close()
