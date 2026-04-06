# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import sys
from pathlib import Path
import matplotlib.gridspec as gridspec

# Get the project root directory
project_root = Path(__file__).resolve().parent.parent  # Adjust if needed
sys.path.append(str(project_root))

# Import from vis_utils
from vis_utils import (
    convert_xzy_hmaps,
    CONTROLLER_PATH_PREFIX,
    CONTROLLER_NAME,
    WORLD_NAME,
    OUTPUT_DIR
)

def load_place_cell_data(scale):
    """
    Load place cell data for a specific scale.
    
    Args:
        scale (int): Scale to load
        
    Returns:
        tuple: (hmap_loc, hmap_pcn)
    """
    import pickle
    import os
    import numpy as np
    
    # Define the path for hmaps
    hmap_directory = os.path.join(
        CONTROLLER_PATH_PREFIX, CONTROLLER_NAME, "pkl", WORLD_NAME, "hmaps"
    )
    
    # Load location data
    hmap_file = "hmap_loc.pkl"
    file_path = os.path.join(hmap_directory, hmap_file)
    with open(file_path, "rb") as f:
        hmap_loc = np.array(pickle.load(f))
        # remove first element
        hmap_loc = hmap_loc[1:]
    print(f"Loaded hmap_loc from {file_path}")
    
    # Load PCN data for the specified scale
    hmap_file = f"hmap_pcn_scale_{scale}.pkl"
    file_path = os.path.join(hmap_directory, hmap_file)
    try:
        with open(file_path, "rb") as f:
            hmap_pcn = np.array(pickle.load(f))
            # remove first element
            hmap_pcn = hmap_pcn[1:]
            print(f"Loaded hmap_pcn_scale_{scale} from {file_path}")
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Cannot process scale {scale}.")
        return None, None
    
    return hmap_loc, hmap_pcn

def analyze_place_cell_aliasing(
    hmap_pcn: np.ndarray,
    hmap_x: np.ndarray,
    hmap_y: np.ndarray,
    scale: int,
    gridsize: int = 100,
    activation_threshold: float = 0.2,
    min_cluster_size: int = 10,
    min_peak_height: float = 0.3,
    min_peak_distance_factor: float = 0.25,  # Minimum distance between peaks as factor of environment size
    large_scale_adjustment: bool = True,
):
    """
    Analyze place cell aliasing by identifying cells with multiple activation peaks.
    
    Args:
        hmap_pcn: Place cell activations (time steps × num_place_cells)
        hmap_x: X coordinates
        hmap_y: Y coordinates
        scale: Current scale being analyzed (used for scale-specific adjustments)
        gridsize: Size of the grid for binning
        activation_threshold: Threshold for considering a cell active
        min_cluster_size: Minimum number of points to form a cluster
        min_peak_height: Minimum peak height relative to max activation
        min_peak_distance_factor: Minimum distance between peaks as fraction of environment size
        large_scale_adjustment: Whether to apply special handling for large scale cells
        
    Returns:
        tuple: (aliased_cells, total_active_cells, aliased_data)
    """
    # Calculate total activation per cell
    total_activation_per_cell = np.sum(hmap_pcn, axis=0)
    
    # Get indices of cells with non-zero activation
    active_cell_indices = np.where(total_activation_per_cell > 0)[0]
    total_active_cells = len(active_cell_indices)
    
    print(f"Found {total_active_cells} active place cells")
    
    # Define the grid boundaries
    xmin, xmax = np.min(hmap_x), np.max(hmap_x)
    ymin, ymax = np.min(hmap_y), np.max(hmap_y)
    
    # Calculate environment size for distance calculations
    env_width = xmax - xmin
    env_height = ymax - ymin
    env_size = np.sqrt(env_width**2 + env_height**2)
    
    # Required minimum distance between peaks (as a fraction of environment size)
    min_peak_distance = env_size * min_peak_distance_factor
    
    # Scale-specific adjustments
    if large_scale_adjustment:
        if scale == 2:  # Large scale
            print(f"Applying large scale adjustments (scale {scale})")
            # Increase minimum peak distance for large scale
            min_peak_distance = env_size * min_peak_distance_factor * 2.0
            # Increase smoothing for large scale
            smoothing_sigma = 2.0
            # Increase local maxima filter size
            local_max_size = 5
        elif scale == 1:  # Medium scale
            smoothing_sigma = 1.5
            local_max_size = 3
        else:  # Small scale
            smoothing_sigma = 1.0
            local_max_size = 3
    else:
        smoothing_sigma = 1.0
        local_max_size = 3
    
    print(f"Minimum peak distance: {min_peak_distance:.2f} units")
    
    # Create grid edges
    xedges = np.linspace(xmin, xmax, gridsize + 1)
    yedges = np.linspace(ymin, ymax, gridsize + 1)
    
    # Prepare list to store aliased cells
    aliased_cells = []
    aliased_data = []
    
    # Analyze each active cell
    for i, cell_index in enumerate(active_cell_indices):
        # Get activations for this cell
        activations = hmap_pcn[:, cell_index]
        
        # Skip cells with weak activation
        if np.max(activations) < activation_threshold:
            continue
        
        # Positions where activation is significant
        mask = activations > activation_threshold * np.max(activations)
        if np.sum(mask) < min_cluster_size:
            continue
            
        x, y, a = hmap_x[mask], hmap_y[mask], activations[mask]
        
        # Create a 2D histogram of activations
        hist, _, _ = np.histogram2d(
            x, y, bins=[xedges, yedges], weights=a
        )
        
        # Smooth the histogram
        hist_smooth = ndimage.gaussian_filter(hist, sigma=smoothing_sigma)
        
        # Find local maxima
        local_max = ndimage.maximum_filter(hist_smooth, size=local_max_size)
        maxima = (hist_smooth == local_max) & (hist_smooth > min_peak_height * np.max(hist_smooth))
        maxima_coords = np.where(maxima)
        
        # If more than one peak is found, this cell may be aliasing
        if len(maxima_coords[0]) > 1:
            # Get coordinates of peaks in original space
            peak_coords = []
            peak_values = []
            for peak_idx in range(len(maxima_coords[0])):
                bin_x = maxima_coords[0][peak_idx]
                bin_y = maxima_coords[1][peak_idx]
                peak_x = xedges[bin_x] + (xedges[bin_x+1] - xedges[bin_x])/2
                peak_y = yedges[bin_y] + (yedges[bin_y+1] - yedges[bin_y])/2
                peak_val = hist_smooth[bin_x, bin_y]
                peak_coords.append((peak_x, peak_y))
                peak_values.append(peak_val)
            
            # Filter peaks that are too close to each other
            filtered_peaks = []
            
            # Sort peaks by activation strength (descending)
            sorted_indices = np.argsort(-np.array(peak_values))
            
            for idx in sorted_indices:
                peak = peak_coords[idx]
                # Check if this peak is far enough from all already accepted peaks
                if not filtered_peaks or all(
                    np.sqrt((peak[0] - p[0])**2 + (peak[1] - p[1])**2) > min_peak_distance 
                    for p in filtered_peaks
                ):
                    filtered_peaks.append(peak)
            
            # Check if we still have multiple peaks after filtering
            if len(filtered_peaks) > 1:
                aliased_cells.append(cell_index)
                aliased_data.append({
                    'cell_index': cell_index,
                    'peak_coords': filtered_peaks,
                    'num_clusters': len(filtered_peaks),
                    'activations': activations,
                    'max_activation': np.max(activations),
                    'x': x,
                    'y': y,
                    'a': a,
                    'hist': hist_smooth,
                })
    
    print(f"Found {len(aliased_cells)} aliased cells out of {total_active_cells} active cells")
    return aliased_cells, total_active_cells, aliased_data

def plot_aliased_cell(
    aliased_info,
    hmap_x,
    hmap_y,
    scale,
    ax=None,
):
    """
    Plot an aliased place cell.
    
    Args:
        aliased_info: Dictionary with aliased cell information
        hmap_x: X coordinates
        hmap_y: Y coordinates
        scale: Scale index
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract cell data
    cell_index = aliased_info['cell_index']
    activations = aliased_info['activations']
    x = aliased_info['x']
    y = aliased_info['y']
    a = aliased_info['a']
    peak_coords = aliased_info['peak_coords']
    
    # Create hexbin plot of cell activations
    hb = ax.hexbin(
        hmap_x,
        hmap_y,
        C=activations,
        gridsize=100,
        reduce_C_function=np.mean,
        cmap='viridis',
        alpha=0.7,
    )
    
    # Mark the peaks
    for i, (px, py) in enumerate(peak_coords):
        ax.plot(px, py, 'ro', markersize=10, markeredgecolor='white')
        ax.text(px, py, str(i+1), color='white', ha='center', va='center', fontweight='bold')
    
    # Add colorbar
    plt.colorbar(hb, ax=ax)
    
    # Set title and labels
    ax.set_title(f"Scale {scale}: Aliased Cell {cell_index}\n{aliased_info['num_clusters']} distinct activation regions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    return ax

def analyze_all_scales(
    scales=None,
    gridsize=100,
    activation_threshold=0.2,
    min_cluster_size=10,
    min_peak_height=0.3,
    min_peak_distance_factor=0.25,
    large_scale_adjustment=True,
    save_path=None,
    show_plot=True,
    max_examples=3,
):
    """
    Analyze place cell aliasing across multiple scales.
    
    Args:
        scales: List of scales to analyze
        gridsize: Size of the grid for binning
        activation_threshold: Threshold for considering a cell active
        min_cluster_size: Minimum number of points to form a cluster
        min_peak_height: Minimum peak height relative to max activation
        min_peak_distance_factor: Minimum distance between peaks as fraction of environment size
        large_scale_adjustment: Whether to apply special handling for large scale cells
        save_path: Path to save the results
        show_plot: Whether to show the plot
        max_examples: Maximum number of example cells to plot per scale
    """
    if scales is None:
        scales = [0, 1, 2]
    
    # Prepare to store results
    results = {}
    
    # Analyze each scale
    for scale in scales:
        print(f"\n==== Analyzing Scale {scale} ====")
        
        # Load data
        hmap_loc, hmap_pcn = load_place_cell_data(scale)
        
        if hmap_loc is None or hmap_pcn is None:
            print(f"Skipping scale {scale} due to missing data")
            continue
        
        # Convert coordinates
        hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc)
        
        # Analyze aliasing
        aliased_cells, total_active_cells, aliased_data = analyze_place_cell_aliasing(
            hmap_pcn=hmap_pcn,
            hmap_x=hmap_x,
            hmap_y=hmap_y,
            scale=scale,
            gridsize=gridsize,
            activation_threshold=activation_threshold,
            min_cluster_size=min_cluster_size,
            min_peak_height=min_peak_height,
            min_peak_distance_factor=min_peak_distance_factor,
            large_scale_adjustment=large_scale_adjustment,
        )
        
        # Store results
        results[scale] = {
            'aliased_cells': aliased_cells,
            'total_active_cells': total_active_cells,
            'percentage': 100 * len(aliased_cells) / total_active_cells if total_active_cells > 0 else 0,
            'aliased_data': aliased_data,
            'hmap_x': hmap_x,
            'hmap_y': hmap_y,
        }
        
        print(f"Scale {scale}: {len(aliased_cells)} aliased cells out of {total_active_cells} active cells")
        print(f"Scale {scale}: {results[scale]['percentage']:.2f}% of cells are aliased")
    
    # Create a plot of example aliased cells and summary statistics
    if show_plot or save_path:
        # Determine how many example cells to show
        num_examples = min(max_examples, max([len(results[s]['aliased_data']) for s in results.keys()], default=0))
        
        if num_examples > 0:
            # Calculate figure dimensions
            fig_height = 8 + 3 * num_examples
            
            # Create figure
            fig = plt.figure(figsize=(15, fig_height))
            
            # Create GridSpec for layout
            gs = gridspec.GridSpec(2 + num_examples, len(scales), height_ratios=[1, 1] + [3] * num_examples)
            
            # Create bar chart of aliasing percentages (top row)
            ax_bar = plt.subplot(gs[0, :])
            scales_list = list(results.keys())
            percentages = [results[s]['percentage'] for s in scales_list]
            
            bars = ax_bar.bar(scales_list, percentages, color='skyblue')
            
            # Add value labels on top of bars
            for bar, percentage in zip(bars, percentages):
                ax_bar.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{percentage:.1f}%",
                    ha='center'
                )
            
            ax_bar.set_xlabel('Scale')
            ax_bar.set_ylabel('Aliased Cells (%)')
            ax_bar.set_title('Percentage of Place Cells with Aliasing')
            ax_bar.set_xticks(scales_list)
            ax_bar.set_xticklabels([f'Scale {s}' for s in scales_list])
            ax_bar.set_ylim(0, max(percentages) * 1.2 if percentages else 10)
            
            # Create a table with the summary statistics (second row)
            ax_table = plt.subplot(gs[1, :])
            ax_table.axis('off')
            
            table_data = [
                [f"{results[s]['aliased_cells']}" for s in scales_list],
                [f"{results[s]['total_active_cells']}" for s in scales_list],
                [f"{results[s]['percentage']:.2f}%" for s in scales_list],
            ]
            
            table = ax_table.table(
                cellText=table_data,
                rowLabels=['Aliased Cells', 'Active Cells', 'Percentage'],
                colLabels=[f'Scale {s}' for s in scales_list],
                loc='center',
                cellLoc='center',
            )
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            
            # Plot example aliased cells for each scale
            for i, scale in enumerate(scales_list):
                aliased_data = results[scale]['aliased_data']
                for j in range(min(num_examples, len(aliased_data))):
                    ax = plt.subplot(gs[2 + j, i])
                    plot_aliased_cell(
                        aliased_info=aliased_data[j],
                        hmap_x=results[scale]['hmap_x'],
                        hmap_y=results[scale]['hmap_y'],
                        scale=scale,
                        ax=ax,
                    )
            
            plt.tight_layout()
            
            # Save the figure if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved analysis results to {save_path}")
            
            # Show the figure if requested
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
    
    return results

# Generate a CSV report of aliasing statistics
def save_aliasing_stats_csv(results, csv_path):
    """
    Save aliasing statistics to a CSV file.
    
    Args:
        results: Dictionary with analysis results
        csv_path: Path to save the CSV file
    """
    import csv
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Scale', 'Total Active Cells', 'Aliased Cells', 'Percentage', 'Average Clusters Per Aliased Cell'])
        
        # Write data for each scale
        for scale in sorted(results.keys()):
            data = results[scale]
            
            # Calculate average number of clusters per aliased cell
            avg_clusters = np.mean([info['num_clusters'] for info in data['aliased_data']]) if data['aliased_data'] else 0
            
            writer.writerow([
                scale,
                data['total_active_cells'],
                len(data['aliased_cells']),
                f"{data['percentage']:.2f}%",
                f"{avg_clusters:.2f}",
            ])
    
    print(f"Saved aliasing statistics to {csv_path}")

# %%
if __name__ == "__main__":
    print("Starting place cell aliasing analysis...")
    
    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, "pcn_aliasing")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the analysis with scale-specific adjustments
    results = analyze_all_scales(
        scales=[0, 1, 2],
        gridsize=100,
        activation_threshold=0.2,
        min_cluster_size=10,
        min_peak_height=0.3,
        min_peak_distance_factor=0.25,  # Peaks must be at least 25% of environment size apart
        large_scale_adjustment=True,    # Apply special handling for large scale cells
        save_path=os.path.join(output_dir, "pcn_aliasing_analysis.png"),
        show_plot=True,
        max_examples=3,
    )
    
    # Save statistics to CSV
    save_aliasing_stats_csv(
        results,
        csv_path=os.path.join(output_dir, "pcn_aliasing_stats.csv"),
    )
    
    print("Place cell aliasing analysis complete!")
