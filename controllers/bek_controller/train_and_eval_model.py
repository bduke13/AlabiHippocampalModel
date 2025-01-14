# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify single GPU to use
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Suppress NUMA warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
from controllers.bek_controller.models.sparse_cnn_v1 import create_models

# Load the preprocessed images
resized_images = np.load("controllers/bek_controller/hmap_images.npy")
print("Processed image shape:", resized_images.shape)

# Enable interactive mode
plt.ion()

# %%
# Preview some random images and their data
plt.figure(figsize=(20, 4))
random_indices = np.random.choice(len(resized_images), 5, replace=False)
for idx, i in enumerate(random_indices):  # Show 5 random images
    plt.subplot(1, 5, idx + 1)
    plt.imshow(resized_images[i])
    plt.axis("off")
    plt.title(f"Image {i}")


# Split the data into training and validation sets
X_train, X_val = train_test_split(resized_images, test_size=0.1, random_state=42)

X_train = X_train / 255.0
X_val = X_val / 255.0


BATCH_SIZE = 4
EPOCHS = 30

# %%

# Train non-sparse model
print("\nTraining Non-Sparse Model...")
encoder, decoder, autoencoder = create_models()
autoencoder.compile(optimizer="adam", loss="mse")
history = autoencoder.fit(
    X_train,
    X_train,  # Input and target are the same for autoencoders
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, X_val),
    shuffle=True,
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
# Print final loss values
print(f"\nFinal training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

# %%
# Save the encoder and decoder separately
encoder.save("encoder_model_std.keras")
decoder.save("decoder_model_std.keras")

# %%
# Load the encoder and decoder models separately
encoder = tf.keras.models.load_model("encoder_model_std.keras")
decoder = tf.keras.models.load_model("decoder_model_std.keras")


# %%
# Get random indices fcr visualization
n = 10  # Number of images to display
random_indices = np.random.choice(len(X_val), n, replace=False)

# Get encoded representations and decoded images for the selected samples
selected_samples = X_val[random_indices]
encoded_output = encoder.predict(selected_samples)
decoded_imgs = decoder.predict(encoded_output)

# Calculate dimensions for reshaping encoded output
encoded_dim = int(np.sqrt(encoded_output.shape[1]))
if encoded_dim * encoded_dim != encoded_output.shape[1]:
    # If not a perfect square, use approximate dimensions
    encoded_dim1 = int(np.sqrt(encoded_output.shape[1]))
    encoded_dim2 = int(np.ceil(encoded_output.shape[1] / encoded_dim1))
else:
    encoded_dim1 = encoded_dim2 = encoded_dim

# Create a figure with 3 rows: original, reconstructed, and encoded
plt.figure(figsize=(20, 12))
for i in range(n):
    # Original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(selected_samples[i])
    plt.title(f"Original {i+1}")
    plt.axis("off")

    # Reconstructed images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title(f"Reconstructed {i+1}")
    plt.axis("off")

    # Encoded representation
    # ax = plt.subplot(3, n, i + 1 + 2 * n)
    # reshaped_encoding = encoded_output[i].reshape(encoded_dim1, encoded_dim2)
    # plt.imshow(reshaped_encoding, cmap="viridis")
    # plt.title(f"Encoded {i+1}")
    # plt.colorbar()
    # plt.axis("off")

plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
import umap
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


# Load the preprocessed images and colors
def load_data():
    images = np.load("controllers/bek_controller/hmap_images.npy")
    with open("controllers/bek_controller/point_colors.pkl", "rb") as f:
        point_colors = np.array(pickle.load(f))
    return images, point_colors


# Normalize images for model input
def normalize_images(images):
    return images / 255.0


# Predict embeddings and reduce dimensions using UMAP
def predict_and_reduce(encoder, images):
    embeddings = encoder.predict(images)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    umap_reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, random_state=42
    )
    umap_embeddings = umap_reducer.fit_transform(embeddings_scaled)
    return umap_embeddings


# Plot UMAP embeddings with colors
def plot_colored_umap(umap_embeddings, point_colors):
    plt.figure(figsize=(10, 10))
    plt.scatter(
        umap_embeddings[:, 0], umap_embeddings[:, 1], c=point_colors, s=5, alpha=0.7
    )
    plt.title("Colored UMAP Embeddings")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.colorbar(label="Color Gradient")
    plt.grid(True)
    plt.show()


# Load the preprocessed data
images, point_colors = load_data()

# Normalize images
normalized_images = normalize_images(images)

# Load the encoder model
encoder = keras.models.load_model("encoder_model_std.keras")

# Predict embeddings and reduce dimensions
umap_embeddings = predict_and_reduce(encoder, normalized_images)

# Plot the UMAP embeddings with colors
plot_colored_umap(umap_embeddings, point_colors)


# ------------------------------------------------------------------------------------------

# %%
# Generate embeddings for all images
embeddings = encoder.predict(resized_images)
print(f"Embedding shape: {embeddings.shape}")

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

# Print how many components are needed for different variance thresholds
thresholds = [0.8, 0.9, 0.95, 0.99]
for threshold in thresholds:
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= threshold) + 1
    print(
        f"Number of components needed for {threshold*100:.0f}% variance: {n_components}"
    )
