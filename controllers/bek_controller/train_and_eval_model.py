# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify single GPU to use
# Suppress NUMA warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.ion()  # Enable interactive mode
from tensorflow.keras import regularizers
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tensorflow.keras.models import save_model

# Load the preprocessed images
resized_images = np.load("preprocessed_images.npy")
print("Processed image shape:", resized_images.shape)

# Preview some images and their data
plt.figure(figsize=(15, 5))
for i in range(3):  # Show first 3 images
    plt.subplot(1, 3, i + 1)
    plt.imshow(resized_images[i])
    plt.axis("off")
    plt.title(f"Image {i}")
#
## Print raw data statistics for each image
# print(f"\nImage {i} statistics:")
# print(f"Shape: {resized_images[i].shape}")
# print(f"Min value: {resized_images[i].min():.3f}")
# print(f"Max value: {resized_images[i].max():.3f}")
# print(f"Mean value: {resized_images[i].mean():.3f}")
# print(f"Standard deviation: {resized_images[i].std():.3f}")


# Split the data into training and validation sets
X_train, X_val = train_test_split(resized_images, test_size=0.1, random_state=42)

# Set batch size for training
BATCH_SIZE = 1


# %%
# Build separate encoder and decoder models
def build_separate_encoder_decoder():
    # Input layer for encoder
    input_img = layers.Input(shape=(96, 192, 3))

    # Encoder
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # (48, 96, 16)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # (24, 48, 32)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # (12, 24, 64)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # (12, 24, 128)

    # Flatten and Dense bottleneck
    shape_before_flattening = x.shape[1:]  # Save the shape for later
    x = layers.Flatten()(x)
    bottleneck = layers.Dense(
        200, activation="relu", activity_regularizer=regularizers.l1(1e-5)
    )(x)

    # Add dropout layer
    # bottleneck = layers.Dropout(0.3)(bottleneck)

    # Encoder model
    encoder = Model(input_img, bottleneck, name="encoder")

    # Decoder input
    encoded_input = layers.Input(shape=(200,))
    x = layers.Dense(np.prod(shape_before_flattening), activation="relu")(encoded_input)
    x = layers.Reshape(shape_before_flattening)(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # (12, 24, 128)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # (24, 48, 64)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # (48, 96, 32)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # (96, 192, 16)

    # Output layer for decoder
    decoded_output = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    # Decoder model
    decoder = Model(encoded_input, decoded_output, name="decoder")

    # Full autoencoder model that connects encoder and decoder
    autoencoder = Model(input_img, decoder(encoder(input_img)))

    return encoder, decoder, autoencoder


# Create the model
encoder, decoder, autoencoder = build_separate_encoder_decoder()
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

# Train the autoencoder
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
history = autoencoder.fit(
    X_train,
    X_train,  # For autoencoders, input = output
    batch_size=BATCH_SIZE,
    epochs=40,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping],
)

# %%
# Save the encoder and decoder separately
encoder.save("encoder_model.keras")
decoder.save("decoder_model.keras")

# %%
# Load the encoder and decoder models separately
encoder = tf.keras.models.load_model("encoder_model.keras")
decoder = tf.keras.models.load_model("decoder_model.keras")

# %%
# Get random indices for visualization
n = 5  # Number of images to display
random_indices = np.random.choice(len(X_val), n, replace=False)

# Get encoded representations and decoded images for the selected samples
selected_samples = X_val[random_indices]
encoded_output = encoder.predict(selected_samples)
decoded_imgs = decoder.predict(encoded_output)

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
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(encoded_output[i].reshape(10, 20), cmap="viridis")
    plt.title(f"Encoded {i+1}")
    plt.colorbar()
    plt.axis("off")

plt.tight_layout()
plt.show()

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

# Print how many components are needed for different variance thresholds
thresholds = [0.8, 0.9, 0.95, 0.99]
for threshold in thresholds:
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= threshold) + 1
    print(
        f"Number of components needed for {threshold*100:.0f}% variance: {n_components}"
    )
