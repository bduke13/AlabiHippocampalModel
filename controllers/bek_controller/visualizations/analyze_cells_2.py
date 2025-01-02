# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle


def calculate_far_cosine_metric(
    positions_x, positions_y, activations, distance_threshold=2.0, n_samples=1000
):
    """
    Calculate the far cosine metric for activation patterns at different locations.

    Args:
        positions_x: Array of x coordinates
        positions_y: Array of y coordinates
        activations: Array of shape [positions, cells] containing activation values
        distance_threshold: Minimum distance to consider points "far apart"
        n_samples: Number of random points to sample (if None, use all points)

    Returns:
        Dictionary containing:
        - far_cosine_values: List of cosine similarities for far pairs
        - far_pair_distances: List of distances for far pairs
        - sampled_indices: Indices of sampled points
    """
    # Sample points if requested
    total_points = len(positions_x)
    if n_samples and n_samples < total_points:
        sample_indices = np.random.choice(total_points, size=n_samples, replace=False)
        positions_x = positions_x[sample_indices]
        positions_y = positions_y[sample_indices]
        activations = activations[sample_indices]
    else:
        sample_indices = np.arange(total_points)

    far_cosine_values = []
    far_pair_distances = []

    # Calculate cosine similarity for all pairs beyond threshold
    for i in range(len(positions_x)):
        for j in range(i + 1, len(positions_x)):
            # Calculate physical distance between points
            dist = np.sqrt(
                (positions_x[i] - positions_x[j]) ** 2
                + (positions_y[i] - positions_y[j]) ** 2
            )

            if dist > distance_threshold:
                # Get full activation vectors for both locations
                loc1_vector = activations[i]
                loc2_vector = activations[j]

                # Skip if either location has no activation
                if np.all(loc1_vector == 0) or np.all(loc2_vector == 0):
                    continue

                # Calculate cosine similarity
                cosine_sim = np.dot(loc1_vector, loc2_vector) / (
                    np.linalg.norm(loc1_vector) * np.linalg.norm(loc2_vector)
                )

                far_cosine_values.append(cosine_sim)
                far_pair_distances.append(dist)

    return {
        "far_cosine_values": far_cosine_values,
        "far_pair_distances": far_pair_distances,
        "sampled_indices": sample_indices,
    }


# %%
prefix = "three_dim/"
# Load data
with open(f"{prefix}hmap_x.pkl", "rb") as f:
    hmap_x = np.array(pickle.load(f))
with open(f"{prefix}hmap_y.pkl", "rb") as f:
    hmap_y = np.array(pickle.load(f))
with open(f"{prefix}hmap_z.pkl", "rb") as f:
    hmap_z = np.asarray(pickle.load(f))


# %%
# Example analysis for one model and angle
wall_angle = 30

# Run analysis
results = calculate_far_cosine_metric(
    positions_x=hmap_x,
    positions_y=hmap_y,
    activations=hmap_z,
    distance_threshold=2.0,
    n_samples=2000,
)

# Print summary statistics
cosine_values = np.array(results["far_cosine_values"])
distances = np.array(results["far_pair_distances"])

print("\nResults Summary:")
print(f"Number of far pairs analyzed: {len(cosine_values)}")
print(f"Average cosine similarity: {np.mean(cosine_values):.3f}")
print(f"Max cosine similarity: {np.max(cosine_values):.3f}")
print(f"Min cosine similarity: {np.min(cosine_values):.3f}")
print(f"Average distance: {np.mean(distances):.3f}")

# Create visualization
plt.figure(figsize=(15, 5))

# Plot 1: Histogram of cosine similarities
# Plot 1: Histogram of non-zero cosine similarities
plt.subplot(121)
nonzero_cosine_values = cosine_values[cosine_values > 0]
plt.hist(nonzero_cosine_values, bins=30, edgecolor="black")
plt.title("Distribution of Non-Zero Cosine Similarities")
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)

# Print additional statistics for non-zero values
print("\nNon-zero cosine similarity statistics:")
print(f"Number of non-zero pairs: {len(nonzero_cosine_values)}")
print(f"Average non-zero cosine similarity: {np.mean(nonzero_cosine_values):.3f}")

# Plot 2: Scatter plot of distance vs cosine similarity
plt.subplot(122)
plt.scatter(distances, cosine_values, alpha=0.5)
plt.title("Distance vs Cosine Similarity")
plt.xlabel("Distance")
plt.ylabel("Cosine Similarity")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Create heatmap of summed far-cosine similarities per location
