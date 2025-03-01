# %%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection


def weighted_mean(data, weights):
    """
    Calculate the weighted mean of `data` using the provided `weights`.
    """
    return np.sum(data * weights) / np.sum(weights)


def compute_place_cell_centers_3d(hmap_pcn, hmap_x, hmap_y, hmap_z):
    """
    Compute the (x, y, z) centers of place fields for each cell.

    Args:
    - hmap_pcn: (num_positions, num_cells) activation map for place cells
    - hmap_x, hmap_y, hmap_z: (num_positions,) each coordinate array

    Returns:
    - centers_3d: (num_cells, 3) array of place cell 3D centers
    """
    num_cells = hmap_pcn.shape[1]
    centers_3d = np.zeros((num_cells, 3))

    for i in range(num_cells):
        activations = hmap_pcn[:, i]
        mask = activations > 0
        if not np.any(mask):
            # No activation for this cell
            centers_3d[i] = (np.nan, np.nan, np.nan)
            continue

        x_mean = weighted_mean(hmap_x[mask], weights=activations[mask])
        y_mean = weighted_mean(hmap_y[mask], weights=activations[mask])
        z_mean = weighted_mean(hmap_z[mask], weights=activations[mask])
        centers_3d[i] = (x_mean, y_mean, z_mean)

    return centers_3d


def plot_pcn_adjacencies_top_percent_3d(
    pcn,
    hmap_x,
    hmap_y,
    hmap_z,
    hmap_pcn,
    top_percent=10.0,
    cell_percent=100.0,
):
    """
    Plots a 3D graph of place cell centers and draws line segments
    between cells with the top `top_percent`% of summed adjacency weights,
    while ignoring cell index 0 and only showing the top `cell_percent`% of cells.

    Args:
    - pcn: Object containing w_rec_tripartite (tensor: shape [n_hd, n_pc, n_pc]).
    - hmap_x, hmap_y, hmap_z: Arrays of x, y, z coordinates (num_positions,).
    - hmap_pcn: Activation map of shape (num_positions, n_pc).
    - top_percent: Percentage threshold for strongest connections to display.
    - cell_percent: Percentage of cells to keep (based on total activation), ignoring cell 0.
    """
    # 1. Compute 3D place cell centers (for all cells)
    place_cell_centers_3d = compute_place_cell_centers_3d(
        hmap_pcn, hmap_x, hmap_y, hmap_z
    )
    num_cells = place_cell_centers_3d.shape[0]

    # 2. Ignore cell index 0 entirely in subsequent calculations
    #    We'll only consider cells in [1, 2, ..., num_cells-1]
    valid_indices = np.arange(1, num_cells)

    # 3. Determine which cells to keep based on cell_percent
    #    We'll compute a total activation measure for each cell (sum of all positions)
    total_activations = np.array([hmap_pcn[:, i].sum() for i in valid_indices])

    # Sort cells by total activation
    # Highest activations come last with argsort, so we'll select from the top
    sorted_order = np.argsort(total_activations)
    # Number of cells to keep
    k = int(np.round((cell_percent / 100.0) * len(valid_indices)))
    k = max(k, 1)  # Ensure we keep at least 1 cell if cell_percent > 0

    # Indices in ascending order, so take the top k from the end
    top_k_local_indices = sorted_order[-k:]
    # Map back to global cell indices
    top_k_global_indices = valid_indices[top_k_local_indices]

    # 4. Extract adjacency by summing over all head directions
    #    shape: (n_hd, n_pc, n_pc) => sum => (n_pc, n_pc)
    w_rec_tripartite = pcn.w_rec_tripartite.cpu().detach().numpy()
    adj_matrix = w_rec_tripartite.sum(axis=0)

    # 5. Determine threshold for the top_percent of adjacency values
    #    but only among the subset of the top_k_global_indices
    #    so we don't let irrelevant (excluded) cells affect threshold
    sub_adj = adj_matrix[np.ix_(top_k_global_indices, top_k_global_indices)]
    flattened = sub_adj.flatten()

    # Filter out zeros
    non_zero_values = flattened[flattened > 0]
    if len(non_zero_values) == 0:
        print("All adjacency weights are zero among the chosen cells. Nothing to plot.")
        return

    threshold = (
        np.percentile(non_zero_values, 100.0 - top_percent)
        if non_zero_values.size > 0
        else 0
    )

    # 6. Build an undirected graph among the chosen top_k_global_indices
    G = nx.Graph()
    # Add nodes with 3D positions
    # We'll skip any that are NaN, but typically if the cell is chosen, it should be valid
    for cell_idx in top_k_global_indices:
        x, y, z = place_cell_centers_3d[cell_idx]
        if not np.isnan(x) and not np.isnan(y) and not np.isnan(z):
            G.add_node(cell_idx, pos=(x, y, z))

    # Add edges if weight >= threshold
    chosen_nodes = list(G.nodes())
    for i in range(len(chosen_nodes)):
        for j in range(i + 1, len(chosen_nodes)):  # undirected => avoid double
            ci = chosen_nodes[i]
            cj = chosen_nodes[j]
            weight = adj_matrix[ci, cj]

            # Must exceed the threshold
            if weight >= threshold and weight > 0.0:
                G.add_edge(ci, cj, weight=weight)

    # 7. Prepare 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(
        f"Top {top_percent}% Adjacencies (among top {cell_percent}% cells, ignoring cell 0)"
    )

    # Node positions
    node_positions = nx.get_node_attributes(G, "pos")
    xs = [node_positions[n][0] for n in G.nodes()]
    ys = [node_positions[n][1] for n in G.nodes()]
    zs = [node_positions[n][2] for n in G.nodes()]

    # Draw nodes (red markers)
    ax.scatter(xs, ys, zs, c="red", marker="o", s=20, alpha=0.8)

    # Get edges and compute widths for optional weighting
    edges = list(G.edges())
    edge_weights = [G[u][v]["weight"] for u, v in edges]
    if edge_weights:
        max_edge_weight = max(edge_weights)
    else:
        max_edge_weight = 1.0

    # Draw edges (all black, scaled by weight if desired)
    for (u, v), w in zip(edges, edge_weights):
        x_vals = [node_positions[u][0], node_positions[v][0]]
        y_vals = [node_positions[u][1], node_positions[v][1]]
        z_vals = [node_positions[u][2], node_positions[v][2]]
        lw = 2.0 * (w / max_edge_weight)  # scale line width
        ax.plot(x_vals, y_vals, z_vals, color="black", linewidth=lw, alpha=0.7)

    # Set axis labels and show
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from visualizations.vis_utils import (
        load_hmaps,
        convert_xzy_hmaps,
        load_layer_pkl,
    )

    prefix = "webots/controllers/flying_3D_pcn/"
    # 1. Load data
    hmap_loc, hmap_pcn_data = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
    hmap_x_data, hmap_z_data, hmap_y_data = convert_xzy_hmaps(hmap_loc)

    # 2. Load PCN (place cell network) object with adjacency
    pcn_data = load_layer_pkl(layer_name="pcn")

    # 3. Plot adjacency:
    #   top_percent=10 means top 10% of adjacency weights
    #   cell_percent=50 means show only the top 50% of cells (by total activation), ignoring index 0
    plot_pcn_adjacencies_top_percent_3d(
        pcn=pcn_data,
        hmap_x=hmap_x_data,
        hmap_y=hmap_y_data,
        hmap_z=hmap_z_data,
        hmap_pcn=hmap_pcn_data,
        top_percent=0.5,
        cell_percent=50.0,
    )
