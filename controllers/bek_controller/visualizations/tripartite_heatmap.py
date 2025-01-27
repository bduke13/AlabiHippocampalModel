# tripartite_heatmap.py

import sys
import os

# Adjust the Python path to include the 'bek_controller' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
bek_controller_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, bek_controller_dir)

import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """Load necessary data for analysis and visualization."""
    # Load the Place Cell Network (PCN)
    with open(os.path.join(bek_controller_dir, "pcn.pkl"), "rb") as f:
        pcn = pickle.load(f)

    return pcn

def main():
    # Load data
    pcn = load_data()

    # Extract tripartite connections
    w_rec_tripartite = pcn.w_rec_tripartite.numpy()  # Convert to NumPy array
    n_hd, num_pc, _ = w_rec_tripartite.shape


    # Aggregate over head directions by computing the average
    adjacency_matrix = np.mean(w_rec_tripartite, axis=0)

    # Normalize the adjacency matrix
    max_weight = np.max(adjacency_matrix)
    if max_weight > 0:
        adjacency_matrix_normalized = adjacency_matrix / max_weight
    else:
        adjacency_matrix_normalized = adjacency_matrix

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(adjacency_matrix_normalized, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Normalized Connection Strength')
    plt.xlabel('Place Cell j')
    plt.ylabel('Place Cell i')
    plt.title('Heatmap of Adjacency Matrix of Tripartite Connections')
    plt.show()

if __name__ == "__main__":
    main()
