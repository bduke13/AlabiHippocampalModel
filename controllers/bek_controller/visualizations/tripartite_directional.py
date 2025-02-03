# tripartite_directional.py

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Adjust the Python path to include the 'bek_controller' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
bek_controller_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, bek_controller_dir)


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
            centers[i] = np.nan, np.nan
            continue
        x_mean = weighted_mean(hmap_x[mask], weights=activations[mask])
        y_mean = weighted_mean(hmap_y[mask], weights=activations[mask])
        centers[i] = x_mean, y_mean

    return centers


def load_data():
    """Load necessary data for analysis and visualization."""
    with open(os.path.join(bek_controller_dir, "pcn.pkl"), "rb") as f:
        pcn = pickle.load(f)

    with open(os.path.join(bek_controller_dir, "hmap_x.pkl"), "rb") as f:
        hmap_x = np.array(pickle.load(f))
    with open(os.path.join(bek_controller_dir, "hmap_y.pkl"), "rb") as f:
        hmap_y = np.array(pickle.load(f))
    with open(os.path.join(bek_controller_dir, "hmap_z.pkl"), "rb") as f:
        hmap_z = np.asarray(pickle.load(f))

    return pcn, hmap_x, hmap_y, hmap_z


def main():
    # Load data
    pcn, hmap_x, hmap_y, hmap_z = load_data()

    # Compute place cell centers
    place_cell_centers = compute_place_cell_centers(hmap_x, hmap_y, hmap_z)

    # Extract tripartite connections
    w_rec_tripartite = pcn.w_rec_tripartite.numpy()
    n_hd, num_pc, _ = w_rec_tripartite.shape

    # Normalize weights per head direction
    max_weight = np.max(w_rec_tripartite)
    w_rec_tripartite_normalized = (
        w_rec_tripartite / max_weight if max_weight > 0 else w_rec_tripartite
    )

    # Define threshold for edges
    non_zero_weights = w_rec_tripartite_normalized[w_rec_tripartite_normalized > 0]
    threshold = np.percentile(non_zero_weights, 99) if non_zero_weights.size > 0 else 0

    # Define colors for head directions
    cmap = plt.cm.get_cmap("hsv", n_hd)
    colors = [cmap(i) for i in range(n_hd)]

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with positions
    for i, (x, y) in enumerate(place_cell_centers):
        if not np.isnan(x) and not np.isnan(y):
            G.add_node(i, pos=(x, y))

    # Add edges with head direction information
    for hd in range(n_hd):
        for i in G.nodes():
            for j in G.nodes():
                weight = w_rec_tripartite_normalized[hd, i, j]
                if weight > threshold:
                    G.add_edge(i, j, color=colors[hd], weight=weight)

    pos = nx.get_node_attributes(G, "pos")

    # Plot
    plt.figure(figsize=(10, 8))

    # Draw nodes and edges with grid enabled
    plt.grid(True)
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="blue")

    edges = G.edges()
    edge_colors = [G[u][v]["color"] for u, v in edges]
    edge_weights = [G[u][v]["weight"] for u, v in edges]
    max_edge_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [w / max_edge_weight * 2 for w in edge_weights]

    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, width=edge_widths, arrows=True, arrowsize=10
    )

    # Add legend for head directions, positioned outside the plot area
    for hd in range(n_hd):
        plt.plot([], [], color=colors[hd], label=f"Head Direction {hd + 1}")
    plt.legend(title="Head Directions", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title("Graph of Place Cell Connections with Head Direction Coloring")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for the legend
    plt.show()


if __name__ == "__main__":
    main()
