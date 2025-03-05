import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from scipy.ndimage import gaussian_filter
from typing import List, Optional, Tuple, Dict
import seaborn as sns

# Import your visualization utilities
from vis_utils import load_hmaps, convert_xzy_hmaps, load_layer_pkl, OUTPUT_DIR

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def verify_grid_influence(grid_influence_value: Optional[float] = None):
    """
    Verify the grid cell integration by analyzing activations and weights
    
    Args:
        grid_influence_value: If provided, displays this value in plot titles
    """
    print("Loading history maps...")
    
    # Load data for both BVC and grid cell layers
    hmap_loc, hmap_pcn, hmap_bvc, hmap_gcn = load_hmaps(
        hmap_names=["hmap_loc", "hmap_pcn", "hmap_bvc", "hmap_gcn"]
    )
    hmap_x, hmap_y, hmap_z = convert_xzy_hmaps(hmap_loc)
    
    # Load network to access weights
    try:
        print("Loading PCN with grid...")
        pcn = load_layer_pkl("pcn_with_grid")
        print("Successfully loaded PCN with grid!")
        
        # Check if it has grid influence parameter
        if hasattr(pcn, 'grid_influence'):
            print(f"Grid influence parameter: {pcn.grid_influence}")
            if grid_influence_value is None:
                grid_influence_value = pcn.grid_influence
    except:
        try:
            print("Loading regular PCN...")
            pcn = load_layer_pkl("pcn")
            print("Loaded regular PCN (without grid)")
            if grid_influence_value is None:
                grid_influence_value = 0.0
        except:
            print("Could not load PCN. Weight analysis will be skipped.")
            pcn = None
    
    # Extract information about the data
    print("\nData sizes:")
    print(f"Location history: {hmap_loc.shape}")
    print(f"Place cell history: {hmap_pcn.shape}")
    print(f"BVC history: {hmap_bvc.shape}")
    print(f"Grid cell history: {hmap_gcn.shape}")
    
    # Run verification analyses
    analyze_activations(hmap_bvc, hmap_gcn, hmap_pcn, grid_influence_value)  # Fixed: Pass hmap_pcn
    analyze_weight_matrices(pcn, grid_influence_value)
    plot_representative_fields(hmap_x, hmap_z, hmap_pcn, hmap_bvc, hmap_gcn, grid_influence_value)
    plot_trajectory_with_fields(hmap_x, hmap_z, hmap_pcn, hmap_bvc, hmap_gcn, grid_influence_value)
    
    print("\nVerification complete! All results saved to:", OUTPUT_DIR)

def analyze_activations(hmap_bvc: np.ndarray, hmap_gcn: np.ndarray, hmap_pcn: np.ndarray,
                       grid_influence: Optional[float] = None):
    """
    Analyze and compare activations between BVC and grid cell networks
    
    Args:
        hmap_bvc: History of BVC activations
        hmap_gcn: History of grid cell activations
        hmap_pcn: History of place cell activations
        grid_influence: Grid influence parameter value
    """
    print("\nAnalyzing cell activations...")
    
    # Calculate basic statistics
    bvc_mean = np.mean(hmap_bvc)
    bvc_max = np.max(hmap_bvc)
    bvc_nonzero = np.mean(hmap_bvc > 0) * 100
    
    gcn_mean = np.mean(hmap_gcn)
    gcn_max = np.max(hmap_gcn)
    gcn_nonzero = np.mean(hmap_gcn > 0) * 100
    
    print(f"BVC activations - Mean: {bvc_mean:.6f}, Max: {bvc_max:.6f}, Non-zero: {bvc_nonzero:.2f}%")
    print(f"Grid activations - Mean: {gcn_mean:.6f}, Max: {gcn_max:.6f}, Non-zero: {gcn_nonzero:.2f}%")
    
    # Calculate mean and max activations per cell
    bvc_mean_per_cell = np.mean(hmap_bvc, axis=0)
    gcn_mean_per_cell = np.mean(hmap_gcn, axis=0)
    
    bvc_max_per_cell = np.max(hmap_bvc, axis=0)
    gcn_max_per_cell = np.max(hmap_gcn, axis=0)
    
    # Create distribution plots
    influence_text = f" (Grid Influence: {grid_influence:.2f})" if grid_influence is not None else ""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Mean activation distributions
    sns.histplot(bvc_mean_per_cell, bins=50, ax=axs[0, 0], color='blue', alpha=0.7)
    axs[0, 0].set_title(f"BVC Mean Activation Distribution{influence_text}")
    axs[0, 0].set_xlabel("Mean Activation")
    axs[0, 0].set_ylabel("Count")
    
    sns.histplot(gcn_mean_per_cell, bins=50, ax=axs[0, 1], color='red', alpha=0.7)
    axs[0, 1].set_title(f"Grid Cell Mean Activation Distribution{influence_text}")
    axs[0, 1].set_xlabel("Mean Activation")
    axs[0, 1].set_ylabel("Count")
    
    # Max activation distributions
    sns.histplot(bvc_max_per_cell, bins=50, ax=axs[1, 0], color='blue', alpha=0.7)
    axs[1, 0].set_title(f"BVC Max Activation Distribution{influence_text}")
    axs[1, 0].set_xlabel("Max Activation")
    axs[1, 0].set_ylabel("Count")
    
    sns.histplot(gcn_max_per_cell, bins=50, ax=axs[1, 1], color='red', alpha=0.7)
    axs[1, 1].set_title(f"Grid Cell Max Activation Distribution{influence_text}")
    axs[1, 1].set_xlabel("Max Activation")
    axs[1, 1].set_ylabel("Count")
    
    plt.tight_layout()
    influence_str = f"_gi{grid_influence:.2f}" if grid_influence is not None else ""
    plt.savefig(os.path.join(OUTPUT_DIR, f"activation_distributions{influence_str}.png"))
    plt.close()
    
    # Activation time series for a few sample cells
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    
    # Place cell activations
    num_samples = 8
    num_steps = min(hmap_bvc.shape[0], 500)  # Limit to 500 steps for clarity
    
    # Select cells with highest max activations
    bvc_indices = np.argsort(bvc_max_per_cell)[-num_samples:]
    gcn_indices = np.argsort(gcn_max_per_cell)[-num_samples:]
    
    # Place some randomly selected cells
    pcn_max_per_cell = np.max(hmap_pcn, axis=0)
    pcn_indices = np.argsort(pcn_max_per_cell)[-num_samples:]  # Use most active cells instead of random
    
    # Plot activations over time
    for i, idx in enumerate(bvc_indices):
        axs[0].plot(np.arange(num_steps), hmap_bvc[:num_steps, idx], 
                   alpha=0.7, label=f"BVC {idx}")
    axs[0].set_title(f"BVC Activations (Sample Cells){influence_text}")
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("Activation")
    axs[0].legend(fontsize='small')
    
    for i, idx in enumerate(gcn_indices):
        axs[1].plot(np.arange(num_steps), hmap_gcn[:num_steps, idx], 
                   alpha=0.7, label=f"Grid {idx}")
    axs[1].set_title(f"Grid Cell Activations (Sample Cells){influence_text}")
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Activation")
    axs[1].legend(fontsize='small')
    
    for i, idx in enumerate(pcn_indices):
        axs[2].plot(np.arange(num_steps), hmap_pcn[:num_steps, idx], 
                   alpha=0.7, label=f"Place {idx}")
    axs[2].set_title(f"Place Cell Activations (Sample Cells){influence_text}")
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Activation")
    axs[2].legend(fontsize='small')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"activation_timeseries{influence_str}.png"))
    plt.close()

def analyze_weight_matrices(pcn, grid_influence: Optional[float] = None):
    """
    Analyze the weight matrices connecting BVCs and grid cells to place cells
    
    Args:
        pcn: Place cell network object
        grid_influence: Grid influence parameter value
    """
    if pcn is None:
        print("No PCN provided. Skipping weight matrix analysis.")
        return
    
    print("\nAnalyzing weight matrices...")
    influence_text = f" (Grid Influence: {grid_influence:.2f})" if grid_influence is not None else ""
    influence_str = f"_gi{grid_influence:.2f}" if grid_influence is not None else ""
    
    # Check if this is the grid-integrated place cell network
    has_grid_weights = hasattr(pcn, 'w_grid') and pcn.w_grid is not None
    
    if not has_grid_weights:
        print("This is a standard PCN without grid weights.")
        
        # Plot only BVC weights
        if hasattr(pcn, 'w_in'):
            w_bvc = pcn.w_in.detach().cpu().numpy() if isinstance(pcn.w_in, torch.Tensor) else pcn.w_in
            
            plt.figure(figsize=(10, 8))
            plt.imshow(w_bvc, aspect='auto', cmap='viridis')
            plt.colorbar(label='Weight Value')
            plt.title(f"BVC → Place Cell Weights{influence_text}")
            plt.xlabel("BVC Index")
            plt.ylabel("Place Cell Index")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"bvc_weights{influence_str}.png"))
            plt.close()
            
            # BVC weight statistics
            print(f"BVC weights - Mean: {np.mean(w_bvc):.6f}, Max: {np.max(w_bvc):.6f}")
            print(f"BVC weights - Non-zero: {np.mean(w_bvc > 0) * 100:.2f}%")
        else:
            print("Could not find BVC weights (w_in).")
    else:
        # We have both types of weights - plot them side by side
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        
        # BVC weights
        if hasattr(pcn, 'w_in'):
            w_bvc = pcn.w_in.detach().cpu().numpy() if isinstance(pcn.w_in, torch.Tensor) else pcn.w_in
            im_bvc = axs[0].imshow(w_bvc, aspect='auto', cmap='viridis')
            axs[0].set_title(f"BVC → Place Cell Weights{influence_text}")
            axs[0].set_xlabel("BVC Index")
            axs[0].set_ylabel("Place Cell Index")
            fig.colorbar(im_bvc, ax=axs[0])
            
            # BVC weight statistics
            print(f"BVC weights - Mean: {np.mean(w_bvc):.6f}, Max: {np.max(w_bvc):.6f}")
            print(f"BVC weights - Non-zero: {np.mean(w_bvc > 0) * 100:.2f}%")
        else:
            axs[0].text(0.5, 0.5, "No BVC weights found", ha='center', va='center')
            axs[0].set_title("BVC → Place Cell Weights (Not Found)")
        
        # Grid weights
        if hasattr(pcn, 'w_grid') and pcn.w_grid is not None:
            w_grid = pcn.w_grid.detach().cpu().numpy() if isinstance(pcn.w_grid, torch.Tensor) else pcn.w_grid
            im_grid = axs[1].imshow(w_grid, aspect='auto', cmap='viridis')
            axs[1].set_title(f"Grid → Place Cell Weights{influence_text}")
            axs[1].set_xlabel("Grid Cell Index")
            axs[1].set_ylabel("Place Cell Index")
            fig.colorbar(im_grid, ax=axs[1])
            
            # Grid weight statistics
            print(f"Grid weights - Mean: {np.mean(w_grid):.6f}, Max: {np.max(w_grid):.6f}")
            print(f"Grid weights - Non-zero: {np.mean(w_grid > 0) * 100:.2f}%")
        else:
            axs[1].text(0.5, 0.5, "No Grid weights found", ha='center', va='center')
            axs[1].set_title("Grid → Place Cell Weights (Not Found)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"weight_matrices{influence_str}.png"))
        plt.close()
        
        # Create histograms of weight distributions
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        
        if hasattr(pcn, 'w_in'):
            w_bvc = pcn.w_in.detach().cpu().numpy() if isinstance(pcn.w_in, torch.Tensor) else pcn.w_in
            sns.histplot(w_bvc.flatten(), bins=50, ax=axs[0], color='blue', alpha=0.7)
            axs[0].set_title(f"BVC Weight Distribution{influence_text}")
            axs[0].set_xlabel("Weight Value")
            axs[0].set_ylabel("Count")
        else:
            axs[0].text(0.5, 0.5, "No BVC weights found", ha='center', va='center')
        
        if hasattr(pcn, 'w_grid') and pcn.w_grid is not None:
            w_grid = pcn.w_grid.detach().cpu().numpy() if isinstance(pcn.w_grid, torch.Tensor) else pcn.w_grid
            sns.histplot(w_grid.flatten(), bins=50, ax=axs[1], color='red', alpha=0.7)
            axs[1].set_title(f"Grid Weight Distribution{influence_text}")
            axs[1].set_xlabel("Weight Value")
            axs[1].set_ylabel("Count")
        else:
            axs[1].text(0.5, 0.5, "No Grid weights found", ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"weight_distributions{influence_str}.png"))
        plt.close()

def plot_representative_fields(hmap_x: np.ndarray, hmap_z: np.ndarray, 
                              hmap_pcn: np.ndarray, hmap_bvc: np.ndarray, 
                              hmap_gcn: np.ndarray, grid_influence: Optional[float] = None):
    """
    Plot representative fields for place cells, BVCs, and grid cells
    
    Args:
        hmap_x, hmap_z: Position history
        hmap_pcn, hmap_bvc, hmap_gcn: Cell activation histories
        grid_influence: Grid influence parameter value
    """
    print("\nPlotting representative fields...")
    influence_text = f" (Grid Influence: {grid_influence:.2f})" if grid_influence is not None else ""
    influence_str = f"_gi{grid_influence:.2f}" if grid_influence is not None else ""
    
    # Function to create field map for a cell
    def create_field_map(x, z, activations, grid_size=50):
        # Determine environment boundaries
        x_min, x_max = np.min(x), np.max(x)
        z_min, z_max = np.min(z), np.max(z)
        
        # Add small buffer
        x_buffer = (x_max - x_min) * 0.05
        z_buffer = (z_max - z_min) * 0.05
        x_min -= x_buffer
        x_max += x_buffer
        z_min -= z_buffer
        z_max += z_buffer
        
        # Create grid for field mapping
        x_edges = np.linspace(x_min, x_max, grid_size + 1)
        z_edges = np.linspace(z_min, z_max, grid_size + 1)
        
        # Create a 2D histogram of cell activations
        cell_map, x_centers, z_centers = np.histogram2d(
            x, z, bins=[x_edges, z_edges], weights=activations
        )
        
        # Create a 2D histogram of visit counts
        visit_counts, _, _ = np.histogram2d(
            x, z, bins=[x_edges, z_edges]
        )
        
        # Avoid division by zero
        visit_counts[visit_counts == 0] = 1
        
        # Normalize by visit counts
        cell_map = cell_map / visit_counts
        
        # Apply Gaussian smoothing
        cell_map = gaussian_filter(cell_map, sigma=1.0)
        
        return cell_map, (x_min, x_max, z_min, z_max)
    
    # Get most active cells for each type
    def get_top_cells(activations, n=8):
        max_per_cell = np.max(activations, axis=0)
        top_indices = np.argsort(max_per_cell)[-n:][::-1]
        return top_indices
    
    # Get representative cells
    place_cells = get_top_cells(hmap_pcn)
    bvc_cells = get_top_cells(hmap_bvc)
    grid_cells = get_top_cells(hmap_gcn)
    
    # Plot place fields
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()
    
    for i, cell_idx in enumerate(place_cells):
        cell_map, (x_min, x_max, z_min, z_max) = create_field_map(
            hmap_x, hmap_z, hmap_pcn[:, cell_idx]
        )
        
        im = axs[i].imshow(cell_map.T, origin='lower', extent=[x_min, x_max, z_min, z_max], 
                         aspect='auto', cmap='viridis')
        axs[i].set_title(f"Place Cell {cell_idx}")
        axs[i].set_xlabel("X Position")
        axs[i].set_ylabel("Z Position")
        axs[i].plot(hmap_x, hmap_z, 'k.', markersize=0.2, alpha=0.3)
        fig.colorbar(im, ax=axs[i])
    
    plt.suptitle(f"Place Cell Fields{influence_text}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"place_fields{influence_str}.png"))
    plt.close()
    
    # Plot BVC fields
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()
    
    for i, cell_idx in enumerate(bvc_cells):
        cell_map, (x_min, x_max, z_min, z_max) = create_field_map(
            hmap_x, hmap_z, hmap_bvc[:, cell_idx]
        )
        
        im = axs[i].imshow(cell_map.T, origin='lower', extent=[x_min, x_max, z_min, z_max], 
                         aspect='auto', cmap='viridis')
        axs[i].set_title(f"BVC {cell_idx}")
        axs[i].set_xlabel("X Position")
        axs[i].set_ylabel("Z Position")
        axs[i].plot(hmap_x, hmap_z, 'k.', markersize=0.2, alpha=0.3)
        fig.colorbar(im, ax=axs[i])
    
    plt.suptitle(f"BVC Fields{influence_text}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"bvc_fields{influence_str}.png"))
    plt.close()
    
    # Plot grid fields
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()
    
    for i, cell_idx in enumerate(grid_cells):
        cell_map, (x_min, x_max, z_min, z_max) = create_field_map(
            hmap_x, hmap_z, hmap_gcn[:, cell_idx]
        )
        
        im = axs[i].imshow(cell_map.T, origin='lower', extent=[x_min, x_max, z_min, z_max], 
                         aspect='auto', cmap='viridis')
        axs[i].set_title(f"Grid Cell {cell_idx}")
        axs[i].set_xlabel("X Position")
        axs[i].set_ylabel("Z Position")
        axs[i].plot(hmap_x, hmap_z, 'k.', markersize=0.2, alpha=0.3)
        fig.colorbar(im, ax=axs[i])
    
    plt.suptitle(f"Grid Cell Fields{influence_text}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"grid_fields{influence_str}.png"))
    plt.close()

def plot_trajectory_with_fields(hmap_x: np.ndarray, hmap_z: np.ndarray, 
                               hmap_pcn: np.ndarray, hmap_bvc: np.ndarray, 
                               hmap_gcn: np.ndarray, grid_influence: Optional[float] = None):
    """
    Plot trajectory with representative fields from each cell type
    
    Args:
        hmap_x, hmap_z: Position history
        hmap_pcn, hmap_bvc, hmap_gcn: Cell activation histories
        grid_influence: Grid influence parameter value
    """
    print("\nPlotting trajectory with fields...")
    influence_text = f" (Grid Influence: {grid_influence:.2f})" if grid_influence is not None else ""
    influence_str = f"_gi{grid_influence:.2f}" if grid_influence is not None else ""
    
    # Get most active cells for each type
    def get_top_cell(activations):
        max_per_cell = np.max(activations, axis=0)
        top_index = np.argmax(max_per_cell)
        return top_index
    
    place_cell = get_top_cell(hmap_pcn)
    bvc_cell = get_top_cell(hmap_bvc)
    grid_cell = get_top_cell(hmap_gcn)
    
    # Create figure with trajectory and activations
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot trajectory
    axs[0, 0].plot(hmap_x, hmap_z, 'k-', linewidth=0.5, alpha=0.7)
    axs[0, 0].set_title(f"Trajectory{influence_text}")
    axs[0, 0].set_xlabel("X Position")
    axs[0, 0].set_ylabel("Z Position")
    axs[0, 0].grid(True)
    
    # Function to create field map
    def create_field_map(x, z, activations, grid_size=50):
        # Determine environment boundaries
        x_min, x_max = np.min(x), np.max(x)
        z_min, z_max = np.min(z), np.max(z)
        
        # Add small buffer
        x_buffer = (x_max - x_min) * 0.05
        z_buffer = (z_max - z_min) * 0.05
        x_min -= x_buffer
        x_max += x_buffer
        z_min -= z_buffer
        z_max += z_buffer
        
        # Create grid for field mapping
        x_edges = np.linspace(x_min, x_max, grid_size + 1)
        z_edges = np.linspace(z_min, z_max, grid_size + 1)
        
        # Create a 2D histogram of cell activations
        cell_map, x_centers, z_centers = np.histogram2d(
            x, z, bins=[x_edges, z_edges], weights=activations
        )
        
        # Create a 2D histogram of visit counts
        visit_counts, _, _ = np.histogram2d(
            x, z, bins=[x_edges, z_edges]
        )
        
        # Avoid division by zero
        visit_counts[visit_counts == 0] = 1
        
        # Normalize by visit counts
        cell_map = cell_map / visit_counts
        
        # Apply Gaussian smoothing
        cell_map = gaussian_filter(cell_map, sigma=1.0)
        
        return cell_map, (x_min, x_max, z_min, z_max)
    
    # Plot place field
    cell_map, (x_min, x_max, z_min, z_max) = create_field_map(
        hmap_x, hmap_z, hmap_pcn[:, place_cell]
    )
    im = axs[0, 1].imshow(cell_map.T, origin='lower', extent=[x_min, x_max, z_min, z_max], 
                     aspect='auto', cmap='viridis')
    axs[0, 1].set_title(f"Place Cell {place_cell}")
    axs[0, 1].set_xlabel("X Position")
    axs[0, 1].set_ylabel("Z Position")
    axs[0, 1].plot(hmap_x, hmap_z, 'k.', markersize=0.2, alpha=0.3)
    fig.colorbar(im, ax=axs[0, 1])
    
    # Plot BVC field
    cell_map, (x_min, x_max, z_min, z_max) = create_field_map(
        hmap_x, hmap_z, hmap_bvc[:, bvc_cell]
    )
    im = axs[1, 0].imshow(cell_map.T, origin='lower', extent=[x_min, x_max, z_min, z_max], 
                     aspect='auto', cmap='viridis')
    axs[1, 0].set_title(f"BVC {bvc_cell}")
    axs[1, 0].set_xlabel("X Position")
    axs[1, 0].set_ylabel("Z Position")
    axs[1, 0].plot(hmap_x, hmap_z, 'k.', markersize=0.2, alpha=0.3)
    fig.colorbar(im, ax=axs[1, 0])
    
    # Plot grid field
    cell_map, (x_min, x_max, z_min, z_max) = create_field_map(
        hmap_x, hmap_z, hmap_gcn[:, grid_cell]
    )
    im = axs[1, 1].imshow(cell_map.T, origin='lower', extent=[x_min, x_max, z_min, z_max], 
                     aspect='auto', cmap='viridis')
    axs[1, 1].set_title(f"Grid Cell {grid_cell}")
    axs[1, 1].set_xlabel("X Position")
    axs[1, 1].set_ylabel("Z Position")
    axs[1, 1].plot(hmap_x, hmap_z, 'k.', markersize=0.2, alpha=0.3)
    fig.colorbar(im, ax=axs[1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"trajectory_with_fields{influence_str}.png"))
    plt.close()

def compare_multiple_grid_influences(influence_values: List[float] = [0.0, 0.5, 1.0], 
                                    world_names: Optional[List[str]] = None):
    """
    Compare place fields across different grid influence values from separate simulations
    
    Args:
        influence_values: List of grid influence values to compare
        world_names: List of world names corresponding to each influence value
                    (if None, uses the current WORLD_NAME for all)
    """
    from vis_utils import WORLD_NAME
    
    if world_names is None:
        # Use the same world name for all
        world_names = [WORLD_NAME] * len(influence_values)
    
    if len(world_names) != len(influence_values):
        print("Error: world_names and influence_values must have the same length")
        return
    
    print(f"\nComparing {len(influence_values)} different grid influence values...")
    
    # Store data for each influence value
    all_fields = []
    
    # Original world name
    original_world_name = WORLD_NAME
    
    # Function to create field map
    def create_field_map(x, z, activations, grid_size=50):
        # Determine environment boundaries
        x_min, x_max = np.min(x), np.max(x)
        z_min, z_max = np.min(z), np.max(z)
        
        # Add small buffer
        x_buffer = (x_max - x_min) * 0.05
        z_buffer = (z_max - z_min) * 0.05
        x_min -= x_buffer
        x_max += x_buffer
        z_min -= z_buffer
        z_max += z_buffer
        
        # Create grid for field mapping
        x_edges = np.linspace(x_min, x_max, grid_size + 1)
        z_edges = np.linspace(z_min, z_max, grid_size + 1)
        
        # Create a 2D histogram of cell activations
        cell_map, x_centers, z_centers = np.histogram2d(
            x, z, bins=[x_edges, z_edges], weights=activations
        )
        
        # Create a 2D histogram of visit counts
        visit_counts, _, _ = np.histogram2d(
            x, z, bins=[x_edges, z_edges]
        )
        
        # Avoid division by zero
        visit_counts[visit_counts == 0] = 1
        
        # Normalize by visit counts
        cell_map = cell_map / visit_counts
        
        # Apply Gaussian smoothing
        cell_map = gaussian_filter(cell_map, sigma=1.0)
        
        return cell_map, (x_min, x_max, z_min, z_max)
    
    # For each influence value
    for world_name, influence in zip(world_names, influence_values):
        # Temporarily set WORLD_NAME
        import vis_utils
        vis_utils.WORLD_NAME = world_name
        
        try:
            # Load data
            hmap_loc, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
            hmap_x, hmap_y, hmap_z = convert_xzy_hmaps(hmap_loc)
            
            # Get most active place cell
            max_activations = np.max(hmap_pcn, axis=0)
            best_cell_idx = np.argmax(max_activations)
            
            # Create field map
            field_map, boundaries = create_field_map(
                hmap_x, hmap_z, hmap_pcn[:, best_cell_idx]
            )
            
            all_fields.append({
                'field_map': field_map,
                'boundaries': boundaries,
                'x': hmap_x,
                'z': hmap_z,
                'influence': influence,
                'world_name': world_name,
                'cell_idx': best_cell_idx
            })
            
            print(f"Loaded data for influence {influence} from world {world_name}")
        except Exception as e:
            print(f"Error loading data for influence {influence} from world {world_name}: {e}")
    
    # Restore original world name
    import vis_utils
    vis_utils.WORLD_NAME = original_world_name
    
    # Plot comparison if we have data
    if len(all_fields) > 0:
        num_columns = min(3, len(all_fields))
        num_rows = (len(all_fields) + num_columns - 1) // num_columns
        
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(5 * num_columns, 5 * num_rows))
        
        # Handle single row or column case
        if num_rows == 1 and num_columns == 1:
            axs = np.array([[axs]])
        elif num_rows == 1:
            axs = axs.reshape(1, -1)
        elif num_columns == 1:
            axs = axs.reshape(-1, 1)
        
        for i, field_data in enumerate(all_fields):
            row = i // num_columns
            col = i % num_columns
            
            field_map = field_data['field_map']
            boundaries = field_data['boundaries']
            x_min, x_max, z_min, z_max = boundaries
            influence = field_data['influence']
            cell_idx = field_data['cell_idx']
            
            im = axs[row, col].imshow(field_map.T, origin='lower', 
                                   extent=[x_min, x_max, z_min, z_max], 
                                   aspect='auto', cmap='viridis')
            axs[row, col].set_title(f"Grid Influence: {influence:.2f} (Cell {cell_idx})")
            axs[row, col].set_xlabel("X Position")
            axs[row, col].set_ylabel("Z Position")
            axs[row, col].plot(field_data['x'], field_data['z'], 'k.', markersize=0.2, alpha=0.3)
            fig.colorbar(im, ax=axs[row, col])
        
        # Hide any unused subplots
        for i in range(len(all_fields), num_rows * num_columns):
            row = i // num_columns
            col = i % num_columns
            axs[row, col].axis('off')
        
        plt.suptitle("Comparison of Place Fields with Different Grid Influence Values")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "grid_influence_comparison.png"))
        plt.close()
        
        print(f"Saved grid influence comparison to {os.path.join(OUTPUT_DIR, 'grid_influence_comparison.png')}")
    else:
        print("No data available for comparison")

if __name__ == "__main__":
    # Run the verification with the current grid influence value
    print("Running verification for the current data...")
    verify_grid_influence(grid_influence_value=1.0)  # Explicitly set the grid influence value to 1.0
    
    # To compare across multiple grid influence values from different simulations:
    # compare_multiple_grid_influences(
    #     influence_values=[0.0, 0.5, 1.0],
    #     world_names=["world_bvc_only", "world_mixed", "world_grid_only"]
    # )