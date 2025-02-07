from calendar import c
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the base directories for "vanilla" and "multiscale" modes
vanilla_base_dir = r"controllers/bek_controller/analysis/stats/vanilla"
multiscale_base_dir = r"controllers/bek_controller/analysis/stats/multiscale"

# Define the worlds to analyze
worlds = ["world0_20x20-2obstacles"]  # For testing

# Load JSON files into a DataFrame
def load_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            # Extract corner label from filename
            try:
                parts = filename.split("_")
                x_coord = int(parts[-2])
                y_coord = int(parts[-1].split(".")[0])
                corner = assign_corner_label(x_coord, y_coord)
            except (IndexError, ValueError):
                print(f"Error parsing corner from filename: {filename}")
                corner = "Unknown"
            
            with open(os.path.join(directory, filename), 'r') as f:
                file_data = json.load(f)
                file_data["corner"] = corner  # Add corner label
                data.append(file_data)
    return pd.DataFrame(data)

# Function to assign corner label based on coordinates
def assign_corner_label(x, y):
    corners = {
        (0, -8): "A",
        (8, -8): "B",
        (8, 0): "C",
    }

    return corners.get((x, y), "Unknown")

# Metrics to compare
metrics = ['total_distance_traveled', 'total_time_secs', 'turn_count']

# Iterate through each world
for world in worlds:
    # Define directories for the current world
    vanilla_dir = os.path.join(vanilla_base_dir, world, "JSON")
    multiscale_dir = os.path.join(multiscale_base_dir, world, "JSON")
    
    # Load data for both modes
    vanilla_df = load_json_files(vanilla_dir)
    multiscale_df = load_json_files(multiscale_dir)
    
    # Compute success rates (success count / total count)
    vanilla_grouped = vanilla_df.groupby('corner')
    multiscale_grouped = multiscale_df.groupby('corner')
    
    vanilla_success_rates = vanilla_grouped['success'].mean()  # Success rate
    vanilla_success_counts = vanilla_grouped['success'].sum()  # Successful runs per corner
    
    multiscale_success_rates = multiscale_grouped['success'].mean()  # Success rate
    multiscale_success_counts = multiscale_grouped['success'].sum()  # Successful runs per corner
    
    # Compute averages and standard errors for all metrics
    vanilla_avg = vanilla_grouped.mean(numeric_only=True)
    multiscale_avg = multiscale_grouped.mean(numeric_only=True)
    
    vanilla_sem = vanilla_grouped.sem(numeric_only=True)
    multiscale_sem = multiscale_grouped.sem(numeric_only=True)
    
    # Replace "success" in averages with calculated success rates
    vanilla_avg['success'] = vanilla_success_rates
    multiscale_avg['success'] = multiscale_success_rates
    
    # Print the tables
    print(f"\nVanilla Averages for {world}:")
    print(vanilla_avg)
    
    print(f"\nMultiscale Averages for {world}:")
    print(multiscale_avg)
    
    print(f"\nVanilla Standard Errors for {world}:")
    print(vanilla_sem)
    
    print(f"\nMultiscale Standard Errors for {world}:")
    print(multiscale_sem)
    
    # Create output directory for saving comparison plots for the current world
    comparison_output_dir = os.path.join(
        "controllers/bek_controller/analysis/stats/comparison",
        world
    )
    os.makedirs(comparison_output_dir, exist_ok=True)
    
    # Plot and save comparison plots for all metrics
    for metric in metrics:
        plt.figure()
        
        # Get all corners present in both modes
        corners_vanilla = vanilla_avg.index.tolist()
        corners_multiscale = multiscale_avg.index.tolist()
        all_corners = sorted(set(corners_vanilla + corners_multiscale))
        
        # Prepare data for plotting
        vanilla_vals = [
            vanilla_avg.loc[corner, metric] if corner in corners_vanilla else np.nan
            for corner in all_corners
        ]
        vanilla_ns = [
            vanilla_success_counts.get(corner, 0) for corner in all_corners
        ]

        multiscale_vals = [
            multiscale_avg.loc[corner, metric] if corner in corners_multiscale else np.nan
            for corner in all_corners
        ]
        multiscale_ns = [
            multiscale_success_counts.get(corner, 0) for corner in all_corners
        ]

        # Replace NaN values in the metric values with zeros for plotting
        vanilla_vals = [0 if np.isnan(val) else val for val in vanilla_vals]
        multiscale_vals = [0 if np.isnan(val) else val for val in multiscale_vals]

        # Define x positions for the bars
        x = np.arange(len(all_corners))  # Number of corners
        width = 0.35

        # Plot "vanilla" metric values
        bars_vanilla = plt.bar(
            x - width / 2, 
            vanilla_vals, 
            label='Vanilla', 
            width=width, 
            alpha=0.7
        )

        # Plot "multiscale" metric values
        bars_multiscale = plt.bar(
            x + width / 2, 
            multiscale_vals, 
            label='Multiscale', 
            width=width, 
            alpha=0.7
        )

        
        # Add labels and legend
        plt.title(f"Comparison of {metric.replace('_', ' ').capitalize()} per Corner ({world})")
        plt.ylabel(metric.replace('_', ' ').capitalize())
        plt.xlabel('Corner')
        plt.xticks(x, all_corners, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(comparison_output_dir, f"comparison_{metric}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved comparison plot for {metric} in {world} to {save_path}.")
