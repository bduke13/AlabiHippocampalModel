# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def weighted_mean_torch(data, weights):
    """Calculate the weighted mean of data using the provided weights."""
    return torch.sum(data * weights) / torch.sum(weights)


def compute_place_cell_centers_torch(hmap_pcn, hmap_x, hmap_y, device):
    """Compute the centers of place fields based on place cell activations using PyTorch."""
    hmap_pcn, hmap_x, hmap_y = (
        hmap_pcn.to(device),
        hmap_x.to(device),
        hmap_y.to(device),
    )
    num_cells = hmap_pcn.shape[1]
    centers = torch.full((num_cells, 2), torch.nan, device=device)

    for i in range(num_cells):
        activations = hmap_pcn[:, i]
        mask = activations > 0
        if not mask.any():
            continue
        x_mean = weighted_mean_torch(hmap_x[mask], weights=activations[mask])
        y_mean = weighted_mean_torch(hmap_y[mask], weights=activations[mask])
        centers[i] = torch.tensor([x_mean, y_mean], device=device)

    return centers.cpu().numpy()


def plot_pcn_adjacencies(pcn, hmap_x, hmap_y, hmap_pcn, percentile=0.03):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute place cell centers
    place_cell_centers = compute_place_cell_centers_torch(
        hmap_pcn=torch.tensor(hmap_pcn, device=device),
        hmap_x=torch.tensor(hmap_x, device=device),
        hmap_y=torch.tensor(hmap_y, device=device),
        device=device,
    )

    # Extract tripartite connections
    w_rec_tripartite = pcn.w_rec_tripartite.to(device)
    n_hd, num_pc, _ = w_rec_tripartite.shape

    # Normalize weights per head direction
    max_weight = torch.max(w_rec_tripartite)
    if max_weight > 0:
        w_rec_tripartite /= max_weight

    # Define threshold for edges
    non_zero_weights = w_rec_tripartite[w_rec_tripartite > 0].detach()
    threshold = (
        torch.quantile(non_zero_weights, percentile)
        if non_zero_weights.numel() > 0
        else torch.tensor(0.0, device=device)
    )

    # Define colors for head directions
    cmap = ("hsv", n_hd)
    colors = [cmap(i) for i in range(n_hd)]

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with positions
    for i, (x, y) in enumerate(place_cell_centers):
        if not np.isnan(x) and not np.isnan(y):
            G.add_node(i, pos=(x, y))

    # Add edges with head direction information
    w_rec_tripartite = w_rec_tripartite.cpu().numpy()
    for hd in range(n_hd):
        for i in G.nodes():
            for j in G.nodes():
                weight = w_rec_tripartite[hd, i, j]
                if weight > threshold.cpu().numpy():
                    G.add_edge(i, j, color=colors[hd], weight=weight)

    pos = nx.get_node_attributes(G, "pos")

    # Plot
    plt.figure(figsize=(10, 8))
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

    # Add legend for head directions
    for hd in range(n_hd):
        plt.plot([], [], color=colors[hd], label=f"Head Direction {hd + 1}")
    plt.legend(title="Head Directions", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title("Graph of Place Cell Connections with Head Direction Coloring")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


if __name__ == "__main__":
    from vis_utils import load_hmaps, convert_xzy_hmaps, load_layer_pkl

    hmap_loc, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn"])
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc=hmap_loc)
    pcn = load_layer_pkl(layer_name="pcn")

    plot_pcn_adjacencies(
        pcn=pcn, hmap_x=hmap_x, hmap_y=hmap_y, hmap_pcn=hmap_pcn, percentile=0.03
    )
