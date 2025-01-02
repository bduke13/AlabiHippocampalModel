# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle


def calculate_far_cosine_metric(
    positions_x, positions_y, activations, distance_threshold=2.0, n_samples=1000
):
    """
    Calculate the far cosine metric for activation patterns at different locations.

    For each sampled location i:
      - Find all other sampled locations j where distance(i, j) > distance_threshold.
      - Compute cosine similarity cos(i, j) for those pairs.
      - Accumulate the sum of these far-cosine similarities for location i (and for j).

    Args:
        positions_x (np.array): Array of x coordinates of shape [n_points].
        positions_y (np.array): Array of y coordinates of shape [n_points].
        activations (np.array): Array of shape [n_points, n_cells] containing activation values.
                               For each point i, activations[i] is the activation vector.
        distance_threshold (float): Minimum distance to consider points "far apart".
        n_samples (int): Number of random points to sample (if None, use all points).

    Returns:
        dict: A dictionary containing:
            - "far_cosine_values": List of all far-cosine similarities found (global list).
            - "far_pair_distances": List of distances for those far pairs (same length as far_cosine_values).
            - "location_cosine_sums": np.array of shape [n_samples], each entry is the sum of far-cosine
                                      similarities for that sampled location.
            - "sampled_indices": Indices of sampled points into the original arrays.
    """

    # Number of total points
    total_points = len(positions_x)

    # Decide whether to sample points
    if (n_samples is not None) and (n_samples < total_points):
        sampled_indices = np.random.choice(total_points, size=n_samples, replace=False)
    else:
        sampled_indices = np.arange(total_points)
        n_samples = total_points  # If you're using all points, override n_samples to total_points

    # Subset the arrays
    sub_x = positions_x[sampled_indices]
    sub_y = positions_y[sampled_indices]
    sub_acts = activations[sampled_indices]

    # Initialize outputs
    far_cosine_values = []
    far_pair_distances = []
    location_cosine_sums = np.zeros(n_samples, dtype=float)

    # Compute pairwise far distances + cosine similarities
    for i in range(n_samples):
        vec_i = sub_acts[i]
        xi, yi = sub_x[i], sub_y[i]

        # Skip locations that are entirely zero (if desired)
        if np.all(vec_i == 0):
            continue

        for j in range(i + 1, n_samples):
            vec_j = sub_acts[j]
            xj, yj = sub_x[j], sub_y[j]

            # Skip if entirely zero
            if np.all(vec_j == 0):
                continue

            # Euclidean distance
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

            if dist > distance_threshold:
                # Cosine similarity
                denom = np.linalg.norm(vec_i) * np.linalg.norm(vec_j)
                if denom == 0:
                    continue  # in case norm is zero
                cosine_sim = np.dot(vec_i, vec_j) / denom

                # Add to global lists
                far_cosine_values.append(cosine_sim)
                far_pair_distances.append(dist)

                # Accumulate sum for both i and j
                location_cosine_sums[i] += cosine_sim
                location_cosine_sums[j] += cosine_sim

    return {
        "far_cosine_values": far_cosine_values,
        "far_pair_distances": far_pair_distances,
        "location_cosine_sums": location_cosine_sums,
        "sampled_indices": sampled_indices,
    }


# %%
# --- Load data ---
prefix = "three_dim/"
with open(f"{prefix}hmap_x.pkl", "rb") as f:
    hmap_x = np.array(pickle.load(f))
with open(f"{prefix}hmap_y.pkl", "rb") as f:
    hmap_y = np.array(pickle.load(f))
with open(f"{prefix}hmap_z.pkl", "rb") as f:
    hmap_z = np.array(pickle.load(f))

# %%
# --- Parameters ---
distance_threshold = 2.0
n_samples = 5000

# --- Analysis ---
results = calculate_far_cosine_metric(
    positions_x=hmap_x,
    positions_y=hmap_y,
    activations=hmap_z,
    distance_threshold=distance_threshold,
    n_samples=n_samples,
)

# --- Extract results ---
cosine_values = np.array(results["far_cosine_values"])
distances = np.array(results["far_pair_distances"])
location_cosine_sums = results["location_cosine_sums"]
sampled_indices = results["sampled_indices"]

# --- Print summary statistics ---
print("\nResults Summary:")
print(f"Total pairs (distance > {distance_threshold}): {len(cosine_values)}")
print(f"Average cosine similarity: {np.mean(cosine_values):.3f}")
print(f"Max cosine similarity: {np.max(cosine_values):.3f}")
print(f"Min cosine similarity: {np.min(cosine_values):.3f}")
print(f"Average distance (for far pairs): {np.mean(distances):.3f}")

# --- Create plots ---
plt.figure(figsize=(15, 5))

# %%
# 1) Histogram of non-zero cosine similarities
plt.subplot(1, 2, 1)
nonzero_cosine_values = cosine_values[cosine_values > 0]
plt.hist(nonzero_cosine_values, bins=30, edgecolor="black")
plt.title("Distribution of Non-Zero Far Cosine Similarities")
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)

print("\nNon-zero cosine similarity statistics:")
print(f"Number of non-zero far pairs: {len(nonzero_cosine_values)}")
if len(nonzero_cosine_values) > 0:
    print(f"Average non-zero cosine similarity: {np.mean(nonzero_cosine_values):.3f}")

# %%
# 2) Scatter plot: distance vs cosine similarity
plt.subplot(1, 2, 2)
plt.scatter(distances, cosine_values, alpha=0.3)
plt.title("Distance vs Cosine Similarity (Far Pairs)")
plt.xlabel("Distance")
plt.ylabel("Cosine Similarity")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- 3) Visualizing the sums as a heatmap over space ---
# Note: We only have the sums for the sampled points. So let's create an array
#       that can hold NaNs for all original points and fill in the sum for
#       sampled points.

# Initialize an array with NaNs to hold the sums for *all* points
heatmap = np.full_like(hmap_x, np.nan, dtype=float)
# Fill the sums for the sampled indices
heatmap[sampled_indices] = location_cosine_sums

plt.figure(figsize=(8, 6))
sc = plt.scatter(hmap_x, hmap_y, c=heatmap, cmap="viridis")
plt.colorbar(sc, label="Sum of Far-Cosine Similarities")
plt.title(f"Heatmap of Total Far-Cosine Similarities")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True, alpha=0.3)
plt.show()
