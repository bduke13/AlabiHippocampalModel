# %%
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns

# Load the encoder model and embeddings
encoder = load_model("encoder_model.keras")
resized_images = np.load("preprocessed_images.npy")
embeddings = encoder.predict(resized_images)

# Standardize the embeddings
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(embeddings)

# 1. K-means clustering
n_clusters = 10  # You can adjust this number
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_embeddings)

# 2. UMAP + HDBSCAN clustering
# First reduce dimensionality with UMAP
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
umap_embeddings = umap_reducer.fit_transform(scaled_embeddings)

# Perform HDBSCAN clustering
hdbscan_clusterer = hdbscan.HDBSCAN(
    min_cluster_size=15, min_samples=5, prediction_data=True
)
hdbscan_labels = hdbscan_clusterer.fit_predict(umap_embeddings)


# Plotting functions
def plot_kmeans_clusters():
    # Use UMAP to visualize K-means clusters in 2D
    umap_viz = umap.UMAP(random_state=42).fit_transform(scaled_embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        umap_viz[:, 0], umap_viz[:, 1], c=kmeans_labels, cmap="tab20", alpha=0.6
    )
    plt.colorbar(scatter)
    plt.title("K-means Clusters Visualized with UMAP")
    plt.savefig("kmeans_clusters.png")
    plt.close()


def plot_hdbscan_clusters():
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=hdbscan_labels,
        cmap="tab20",
        alpha=0.6,
    )
    plt.colorbar(scatter)
    plt.title("HDBSCAN Clusters")
    plt.savefig("hdbscan_clusters.png")
    plt.close()


# Generate plots
plot_kmeans_clusters()
plot_hdbscan_clusters()

# Print clustering statistics
print(f"\nK-means found {n_clusters} clusters")
print(f"HDBSCAN found {len(np.unique(hdbscan_labels))} clusters")
print(f"Number of noise points in HDBSCAN: {np.sum(hdbscan_labels == -1)}")

# Save cluster assignments
np.save("kmeans_labels.npy", kmeans_labels)
np.save("hdbscan_labels.npy", hdbscan_labels)
