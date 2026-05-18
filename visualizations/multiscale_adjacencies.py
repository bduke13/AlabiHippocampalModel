import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def plot_w_in_adjacency(pcn, hmap_x, hmap_y, percentile=0.03):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract w_in matrix and determine valid range
    w_in = pcn.w_in.to(device)
    num_source, num_target = w_in.shape  # Shape (2000, 800)

    # Adjust node count based on target cells (800) and available coordinates
    num_coords = min(num_target, len(hmap_x), len(hmap_y))

    print(f"w_in shape: {w_in.shape}, hmap_x: {len(hmap_x)}, hmap_y: {len(hmap_y)}")
    print(f"Using {num_source} source cells and {num_coords} target cells.")

    # Normalize weights
    max_weight = torch.max(w_in)
    if max_weight > 0:
        w_in /= max_weight

    # Determine threshold based on percentile
    non_zero_weights = w_in[w_in > 0].detach()
    threshold = (
        torch.quantile(non_zero_weights, percentile)
        if non_zero_weights.numel() > 0
        else torch.tensor(0.0, device=device)
    )

    # Create graph
    G = nx.DiGraph()

    print(f"Adding {num_coords} target nodes to the graph...")
    for i in range(num_coords):
        G.add_node(i, pos=(hmap_x[i], hmap_y[i]))

    # Assign dummy positions to source nodes
    print(f"Adding {num_source} source nodes to the graph...")
    for i in range(num_source):
        # Assign positions outside the range of target nodes
        G.add_node(num_coords + i, pos=(hmap_x[-1] + i * 10, hmap_y[-1] + i * 10))

    print("Adding edges based on w_in matrix...")
    w_in = w_in.cpu().numpy()

    # Iterate only within valid ranges
    for i in tqdm(range(num_source), desc="Processing source cells"):
        for j in range(num_coords):  # Limit to num_coords target cells
            if j >= num_target:
                continue
            weight = w_in[i, j]
            if weight > threshold.cpu().numpy():
                G.add_edge(num_coords + i, j, weight=weight)  # Connect source to target

    # Get positions for nodes that exist in the graph
    pos = nx.get_node_attributes(G, "pos")

    # Verify that all nodes in G have positions
    missing_nodes = [node for node in G.nodes if node not in pos]
    if missing_nodes:
        print(f"Warning: Nodes {missing_nodes} are missing positions and will be skipped.")

    # Plot
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color="blue")

    edges = G.edges()
    edge_weights = [G[u][v]["weight"] for u, v in edges]
    max_edge_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [w / max_edge_weight * 2 for w in edge_weights]

    print("Plotting adjacency matrix...")
    edges = list(G.edges())
    edge_weights = [G[u][v]["weight"] for u, v in edges]
    max_edge_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [w / max_edge_weight * 2 for w in edge_weights]

    print("Plotting adjacency matrix...")
    for u, v, width in tqdm(zip(edges, edge_widths), desc="Drawing edges", total=len(edges)):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, arrows=True, arrowsize=8, edge_color="gray")

    plt.title(f"w_in Adjacency Matrix (Top {percentile*100:.1f}% of connections)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
    print("Done.")

if __name__ == "__main__":
    from visualizations.vis_utils import load_hmaps, convert_xzy_hmaps, load_layer_pkl

    # Load data
    hmap_loc, hmap_pcn = load_hmaps(hmap_names=["hmap_loc", "hmap_pcn_scale_0"])
    hmap_x, hmap_z, hmap_y = convert_xzy_hmaps(hmap_loc=hmap_loc)
    pcn = load_layer_pkl(layer_name="pcn_scale_0")

    # Plot adjacency matrix with user-defined threshold
    plot_w_in_adjacency(pcn=pcn, hmap_x=hmap_x, hmap_y=hmap_y, percentile=0.01)