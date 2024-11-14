# %%
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model
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

# %%
# Print how many components are needed for different variance thresholds
thresholds = [0.8, 0.9, 0.95, 0.99]
for threshold in thresholds:
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= threshold) + 1
    print(
        f"Number of components needed for {threshold*100:.0f}% variance: {n_components}"
    )

# %%
# Analyze similarity patterns between embedding dimensions
plt.figure(figsize=(15, 10))

# 1. Correlation matrix between embedding dimensions
plt.subplot(121)
correlation_matrix = np.corrcoef(embeddings.T)
sns.heatmap(
    correlation_matrix, cmap="RdBu_r", center=0, xticklabels=False, yticklabels=False
)
plt.title("Correlation Matrix Between\nEmbedding Dimensions")
plt.xlabel("Dimension Index")
plt.ylabel("Dimension Index")

# 2. Distribution of correlation coefficients
plt.subplot(122)
# Get upper triangle of correlation matrix (excluding diagonal)
upper_tri = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
plt.hist(upper_tri, bins=50, density=True)
plt.title("Distribution of Pairwise\nDimension Correlations")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Density")

plt.tight_layout()
plt.show()

# Print summary statistics of correlations
print("\nCorrelation Statistics:")
print(f"Mean correlation: {np.mean(upper_tri):.4f}")
print(f"Std of correlations: {np.std(upper_tri):.4f}")
print(f"Min correlation: {np.min(upper_tri):.4f}")
print(f"Max correlation: {np.max(upper_tri):.4f}")

# %%
# Calculate and visualize cosine similarities between embeddings
from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity for a random subset of embeddings
sample_size = min(1000, embeddings.shape[0])
random_indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
sample_embeddings = embeddings[random_indices]
cosine_sim = cosine_similarity(sample_embeddings)

plt.figure(figsize=(15, 5))

# 1. Heatmap of cosine similarities
plt.subplot(121)
sns.heatmap(cosine_sim, cmap="viridis", xticklabels=False, yticklabels=False)
plt.title("Cosine Similarity Heatmap\n(Random Sample)")
plt.xlabel("Sample Index")
plt.ylabel("Sample Index")

# 2. Distribution of cosine similarities
plt.subplot(122)
# Get upper triangle (excluding diagonal)
upper_tri_cos = cosine_sim[np.triu_indices_from(cosine_sim, k=1)]
plt.hist(upper_tri_cos, bins=50, density=True)
plt.title("Distribution of Pairwise\nCosine Similarities")
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")

plt.tight_layout()
plt.show()

# Print summary statistics of cosine similarities
print("\nCosine Similarity Statistics:")
print(f"Mean similarity: {np.mean(upper_tri_cos):.4f}")
print(f"Std of similarities: {np.std(upper_tri_cos):.4f}")
print(f"Min similarity: {np.min(upper_tri_cos):.4f}")
print(f"Max similarity: {np.max(upper_tri_cos):.4f}")
