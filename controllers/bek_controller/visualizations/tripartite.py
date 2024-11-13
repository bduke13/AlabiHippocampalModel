# tripartite.py

import sys
import os

# Adjust the Python path to include the 'bek_controller' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
bek_controller_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, bek_controller_dir)

import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from numpy.random import default_rng

def weighted_mean(data, weights):
    """Calculate the weighted mean of data using the provided weights."""
    return np.sum(data * weights) / np.sum(weights)

def compute_place_cell_centers(hmap_x, hmap_y, hmap_z):
    """Compute the centers of place fields based on place cell activations."""
    num_cells = hmap_z.shape[1]
    centers = np.zeros((num_cells, 2))

    for i in range(num_cells):
        activations = hmap_z[:, i]
        mask = activations > 0
        if not np.any(mask):
            # If the cell has no activations, set its center to NaN
            centers[i] = np.nan, np.nan
            continue
        x_mean = weighted_mean(hmap_x[mask], weights=activations[mask])
        y_mean = weighted_mean(hmap_y[mask], weights=activations[mask])
        centers[i] = x_mean, y_mean

    return centers

def load_data():
    """Load necessary data for analysis and visualization."""
    # Load the Place Cell Network (PCN)
    with open("C:/Users/alexm/Documents/senior_design/AlabiHippocampalModel-1/controllers/bek_controller/pcn.pkl", "rb") as f:
        pcn = pickle.load(f)

    # Load hmap data
    with open("C:/Users/alexm/Documents/senior_design/AlabiHippocampalModel-1/controllers/bek_controller/hmap_x.pkl", "rb") as f:
        hmap_x = np.array(pickle.load(f))
    with open("C:/Users/alexm/Documents/senior_design/AlabiHippocampalModel-1/controllers/bek_controller/hmap_y.pkl", "rb") as f:
        hmap_y = np.array(pickle.load(f))
    with open("C:/Users/alexm/Documents/senior_design/AlabiHippocampalModel-1/controllers/bek_controller/hmap_z.pkl", "rb") as f:
        hmap_z = np.asarray(pickle.load(f))

    return pcn, hmap_x, hmap_y, hmap_z

def main():
    # Load data
    pcn, hmap_x, hmap_y, hmap_z = load_data()

    # Compute place cell centers
    place_cell_centers = compute_place_cell_centers(hmap_x, hmap_y, hmap_z)

    # Extract tripartite connections and create adjacency matrix
    w_rec_tripartite = pcn.w_rec_tripartite.numpy()  # Convert to NumPy array
    n_hd, num_pc, _ = w_rec_tripartite.shape

    # Aggregate over head directions by computing the average
    adjacency_matrix = np.mean(w_rec_tripartite, axis=0)

    # Normalize the adjacency matrix for consistent thresholding
    max_weight = np.max(adjacency_matrix)
    if max_weight > 0:
        adjacency_matrix = adjacency_matrix / max_weight
    else:
        print("Warning: All weights are zero.")
        adjacency_matrix = adjacency_matrix

    # Define a threshold to include edges (e.g., include top 5% strongest connections)
    non_zero_weights = adjacency_matrix[adjacency_matrix > 0]
    if non_zero_weights.size > 0:
        threshold = np.percentile(non_zero_weights, 95)
    else:
        threshold = 0

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with positions
    for i, (x, y) in enumerate(place_cell_centers):
        if not np.isnan(x) and not np.isnan(y):
            G.add_node(i, pos=(x, y))

    # Add edges based on the adjacency matrix
    for i in G.nodes():
        for j in G.nodes():
            weight = adjacency_matrix[i, j]
            if weight > threshold:
                G.add_edge(i, j)

    # Get positions for nodes
    pos = nx.get_node_attributes(G, 'pos')

    # Draw the graph
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=10)
    plt.title('Graph Visualization of Place Cell Connections')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
