# %%
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import seaborn as sns

# Load the encoder model
encoder = load_model("encoder_model.keras")

# Load preprocessed images
resized_images = np.load("preprocessed_images.npy")

# %%
# Generate embeddings for all images
embeddings = encoder.predict(resized_images)
print(f"Embedding shape: {embeddings.shape}")

# %%
# Calculate basic statistics
mean_activations = np.mean(embeddings, axis=0)
std_activations = np.std(embeddings, axis=0)
max_activations = np.max(embeddings, axis=0)
sparsity = np.mean(embeddings == 0)

print("\nEmbedding Statistics:")
print(f"Mean activation: {np.mean(mean_activations):.4f}")
print(f"Std of activations: {np.mean(std_activations):.4f}")
print(f"Max activation: {np.max(max_activations):.4f}")
print(f"Sparsity (fraction of zeros): {sparsity:.4f}")

# %%
# Plot distribution of activations
plt.figure(figsize=(15, 5))

# 1. Distribution of mean activations across neurons
plt.subplot(131)
plt.hist(mean_activations, bins=50)
plt.title("Distribution of Mean Activations\nAcross Neurons")
plt.xlabel("Mean Activation")
plt.ylabel("Count")

# 2. Distribution of standard deviations
plt.subplot(132)
plt.hist(std_activations, bins=50)
plt.title("Distribution of Standard Deviations\nAcross Neurons")
plt.xlabel("Standard Deviation")
plt.ylabel("Count")

# 3. Heatmap of random subset of embeddings
plt.subplot(133)
sample_size = min(100, embeddings.shape[0])
random_indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
sample_embeddings = embeddings[random_indices]
sns.heatmap(sample_embeddings.T, cmap="viridis")
plt.title("Heatmap of Random Sample\nEmbeddings")
plt.xlabel("Sample Index")
plt.ylabel("Neuron Index")

plt.tight_layout()
plt.show()

# %%
# Additional analysis: Principal Component Analysis
from sklearn.decomposition import PCA

pca = PCA()
pca_result = pca.fit_transform(embeddings)

# Plot explained variance ratio
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.title("PCA Analysis of Embeddings")
plt.grid(True)
plt.show()

# Print how many components are needed for different variance thresholds
thresholds = [0.8, 0.9, 0.95, 0.99]
for threshold in thresholds:
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= threshold) + 1
    print(
        f"Number of components needed for {threshold*100:.0f}% variance: {n_components}"
    )
