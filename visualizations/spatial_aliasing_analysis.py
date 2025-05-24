# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
import time

# Get the project root directory
project_root = Path(__file__).resolve().parent.parent  # Adjust if needed
sys.path.append(str(project_root))

# Import from vis_utils
from vis_utils import (
    load_multi_scale_hmaps,
    convert_xzy_hmaps,
    OUTPUT_DIR
)

# ===================== CONFIGURABLE PARAMETERS =====================
# Spatial Aliasing parameters
DISTANCE_THRESHOLD = 2.0  # Minimum distance (meters) to consider for spatial aliasing
GRID_SIZE = 50  # Size of hexbin grid for discretizing the environment
FILTER_BOTTOM_RATIO = 0.1  # Fraction of lowest activations to filter out
SCALES_TO_ANALYZE = [0, 1, 2]  # Scales to analyze (0=small, 1=medium, 2=large)
SCALE_NAMES = ["Small", "Medium", "Large"]
CELLS_TO_SAMPLE = None  # Number of cells to sample per scale (set to None for all)

# Output directory
OUTPUT_DIR_ALIASING = os.path.join(OUTPUT_DIR, "spatial_aliasing")
# =====================================================================

def create_hexbin(
    cell_index,
    hmap_x,
    hmap_y,
    hmap_pcn,
    gridsize=GRID_SIZE,
    normalize=True,
    filter_bottom_ratio=FILTER_BOTTOM_RATIO,
    analyze=True,
    close_plot=True
):
    """
    Creates a hexbin plot for a given place cell index.
    
    Args:
        cell_index: Index of the place cell to plot
        hmap_x: X coordinates
        hmap_y: Y coordinates
        hmap_pcn: Place cell activations
        gridsize: Size of the hexbin grid
        normalize: Whether to normalize activations
        filter_bottom_ratio: Fraction of lowest activations to filter out
        analyze: Whether to return data for analysis
        close_plot: Whether to close the plot after creation
        
    Returns:
        fig: Figure object
        ax: Axis object
        hb: Hexbin object
        binned_data: List of (x, y, activation) tuples if analyze=True
    """
    # Get activations for this cell
    activations = hmap_pcn[:, cell_index]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create hexbin
    hb = ax.hexbin(
        hmap_x,
        hmap_y,
        C=activations,
        gridsize=gridsize,
        reduce_C_function=np.mean,
        cmap='viridis',
        edgecolors="none",
    )
    
    # Get aggregated activations per bin
    counts = hb.get_array()
    
    # Filter low activations if requested
    if filter_bottom_ratio > 0.0:
        counts_array = np.asarray(counts)
        positive_counts = counts_array[counts_array > 0]
        
        # Only apply threshold if we have positive values
        if len(positive_counts) > 0:
            threshold = np.percentile(positive_counts, filter_bottom_ratio * 100)
            counts[counts < threshold] = 0.0
    
    # Normalize if requested
    if normalize:
        max_count = counts.max()
        if max_count > 0:
            counts = counts / max_count
    
    # Set the values back to the hexbin
    hb.set_array(counts)
    
    # Get bin centers and values if analyze is True
    if analyze:
        offsets = hb.get_offsets()
        x_centers = offsets[:, 0]
        y_centers = offsets[:, 1]
        counts = hb.get_array()
        binned_data = [
            (x, y, activation) for x, y, activation in zip(x_centers, y_centers, counts)
            if activation > 0  # Only include bins with positive activation
        ]
        
        if close_plot:
            plt.close(fig)
        
        return fig, ax, hb, binned_data
    
    if close_plot:
        plt.close(fig)
    
    return fig, ax, hb, counts

def stack_binned_data_by_location(list_of_binned_data):
    """
    Stack binned data from all place cells into a dictionary where bins
    are identified by their center coordinates.
    
    Args:
        list_of_binned_data: List of binned data from multiple place cells
        
    Returns:
        stacked_dict: Dictionary mapping (x,y) coordinates to lists of activations
    """
    stacked_dict = defaultdict(list)
    for binned_data in list_of_binned_data:
        for x, y, act in binned_data:
            # Round coordinates to reduce floating point issues
            key = (round(x, 6), round(y, 6))
            stacked_dict[key].append(act)
    return stacked_dict

def compute_cosine_similarity_sums(stacked_dict, distance_threshold=DISTANCE_THRESHOLD):
    """
    Compute the sum of cosine similarities between bins in the environment,
    considering only bins that are more than a specified distance apart.
    
    Args:
        stacked_dict: Dictionary mapping (x,y) coordinates to lists of activations
        distance_threshold: Minimum distance to consider for similarity calculation
        
    Returns:
        similarity_sums: Dictionary mapping (x,y) coordinates to sum of similarities
    """
    from numpy.linalg import norm
    
    coords = np.array(list(stacked_dict.keys()))
    similarity_sums = {}
    
    # For each bin
    for i, (x_i, y_i) in enumerate(coords):
        vec_i = np.array(stacked_dict[(x_i, y_i)])
        sum_similarity = 0.0
        count = 0
        
        # Compare with all other bins
        for j, (x_j, y_j) in enumerate(coords):
            if i == j:
                continue
            
            # Calculate Euclidean distance
            distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
            
            # Only consider distant bins
            if distance > distance_threshold:
                vec_j = np.array(stacked_dict[(x_j, y_j)])
                
                # Ensure both vectors are non-zero
                if len(vec_i) > 0 and len(vec_j) > 0 and norm(vec_i) > 0 and norm(vec_j) > 0:
                    # If vectors have different lengths, pad the shorter one
                    if len(vec_i) != len(vec_j):
                        if len(vec_i) < len(vec_j):
                            vec_i = np.pad(vec_i, (0, len(vec_j) - len(vec_i)))
                        else:
                            vec_j = np.pad(vec_j, (0, len(vec_i) - len(vec_j)))
                    
                    # Compute cosine similarity
                    cos_sim = np.dot(vec_i, vec_j) / (norm(vec_i) * norm(vec_j))
                    sum_similarity += cos_sim
                    count += 1
        
        # Store the average similarity for this bin
        if count > 0:
            similarity_sums[(x_i, y_i)] = sum_similarity / count
        else:
            similarity_sums[(x_i, y_i)] = 0.0
    
    return similarity_sums

def analyze_scale(scale_idx, scale_name, verbose=True):
    """
    Analyze a single scale for spatial aliasing.
    
    Args:
        scale_idx: Index of the scale to analyze
        scale_name: Name of the scale for display
        verbose: Whether to print progress information
        
    Returns:
        sai_dict: Dictionary mapping (x,y) coordinates to SAI values
        msai: Mean Spatial Aliasing Index
        coords: Array of bin coordinates
        sai_values: Array of SAI values
    """
    if verbose:
        print(f"\n==== Analyzing {scale_name} (Scale {scale_idx}) ====")
    
    # Load data for this scale
    hmap_loc, hmap_pcn_dict = load_multi_scale_hmaps(scales=[scale_idx])
    if scale_idx not in hmap_pcn_dict:
        print(f"Error: No data found for scale {scale_idx}")
        return {}, 0.0, np.array([]), np.array([])
    
    # Extract data
    hmap_pcn = hmap_pcn_dict[scale_idx]
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
    
    if verbose:
        print(f"Data loaded: hmap_pcn shape = {hmap_pcn.shape}")
    
    # Sample cells if specified
    num_cells = hmap_pcn.shape[1]
    if CELLS_TO_SAMPLE is not None and CELLS_TO_SAMPLE < num_cells:
        cell_indices = np.random.choice(num_cells, CELLS_TO_SAMPLE, replace=False)
        if verbose:
            print(f"Sampling {CELLS_TO_SAMPLE} cells out of {num_cells}")
    else:
        cell_indices = np.arange(num_cells)
        if verbose:
            print(f"Using all {num_cells} cells")
    
    # Process cells to get binned data
    all_binned_data = []
    start_time = time.time()
    
    for i, cell_idx in enumerate(cell_indices):
        if verbose and i % 50 == 0:
            elapsed = time.time() - start_time
            if i > 0:
                cells_per_sec = i / elapsed
                remaining = len(cell_indices) - i
                eta = remaining / cells_per_sec if cells_per_sec > 0 else 0
                print(f"Processing cell {i}/{len(cell_indices)} - {cells_per_sec:.1f} cells/sec, ETA: {eta:.1f}s")
        
        try:
            # Create hexbin for this cell
            _, _, _, binned_data = create_hexbin(
                cell_index=cell_idx,
                hmap_x=hmap_x,
                hmap_y=hmap_y,
                hmap_pcn=hmap_pcn,
                gridsize=GRID_SIZE,
                normalize=True,
                filter_bottom_ratio=FILTER_BOTTOM_RATIO,
                analyze=True,
                close_plot=True
            )
            
            # Add to list if not empty
            if binned_data:
                all_binned_data.append(binned_data)
        except Exception as e:
            if verbose:
                print(f"Error processing cell {cell_idx}: {e}")
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"Processed {len(cell_indices)} cells in {elapsed:.1f}s ({len(cell_indices)/elapsed:.1f} cells/sec)")
        print(f"Found data for {len(all_binned_data)} cells")
    
    if not all_binned_data:
        print(f"No valid data found for scale {scale_idx}")
        return {}, 0.0, np.array([]), np.array([])
    
    # Stack binned data by location
    if verbose:
        print("Stacking binned data by location...")
    
    stacked_dict = stack_binned_data_by_location(all_binned_data)
    
    if verbose:
        print(f"Found {len(stacked_dict)} unique spatial bins")
    
    # Compute cosine similarity sums (the Spatial Aliasing Index)
    if verbose:
        print("Computing spatial aliasing indices...")
    
    sai_dict = compute_cosine_similarity_sums(stacked_dict, distance_threshold=DISTANCE_THRESHOLD)
    
    # Calculate Mean Spatial Aliasing Index (MSAI)
    sai_values = np.array(list(sai_dict.values()))
    msai = np.mean(sai_values)
    
    if verbose:
        print(f"Mean Spatial Aliasing Index (MSAI): {msai:.6f}")
    
    # Prepare data for visualization
    coords = np.array(list(sai_dict.keys()))
    
    return sai_dict, msai, coords, sai_values

def run_analysis():
    """
    Run the spatial aliasing analysis for all scales.
    
    Returns:
        results_df: DataFrame with MSAI results for all scales
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR_ALIASING, exist_ok=True)
    
    # Analyze each scale
    results = []
    
    for i, scale_idx in enumerate(SCALES_TO_ANALYZE):
        scale_name = SCALE_NAMES[i] if i < len(SCALE_NAMES) else f"Scale {scale_idx}"
        
        # Analyze this scale
        sai_dict, msai, coords, sai_values = analyze_scale(
            scale_idx=scale_idx,
            scale_name=scale_name,
            verbose=True
        )
        
        if len(coords) > 0:
            # Save results
            scale_dir = os.path.join(OUTPUT_DIR_ALIASING, f"scale_{scale_idx}")
            os.makedirs(scale_dir, exist_ok=True)
            
            np.save(os.path.join(scale_dir, "coords.npy"), coords)
            np.save(os.path.join(scale_dir, "sai_values.npy"), sai_values)
            
            # Add to results list
            results.append({
                "scale_idx": scale_idx,
                "scale_name": scale_name,
                "msai": msai,
                "num_bins": len(sai_dict),
                "max_sai": np.max(sai_values),
                "min_sai": np.min(sai_values)
            })
            
            print(f"Results saved to {scale_dir}")
    
    # Create and save summary DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR_ALIASING, "msai_results.csv"), index=False)
    
    print("\nSummary of Mean Spatial Aliasing Index (MSAI):")
    if not results_df.empty:
        print(results_df[["scale_name", "msai", "num_bins"]].to_string(index=False))
    else:
        print("No valid results found for any scale.")
    
    return results_df

# Execute the analysis when run as a script
if __name__ == "__main__":
    print("Starting Spatial Aliasing Analysis...")
    results_df = run_analysis()
    print("Analysis complete!")