# %%
# autoencoder_test.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved encoder and decoder models
encoder = load_model("encoder_model.keras")
decoder = load_model("decoder_model.keras")

# Load preprocessed images
preprocessed_images = np.load("preprocessed_images.npy")

# Select a subset of images for testing
n = 5
test_images = preprocessed_images[:n]

# Generate bottleneck features using the encoder
bottleneck_features = encoder.predict(test_images)
print("Bottleneck feature shape:", bottleneck_features.shape)

# Reconstruct the images using the decoder
reconstructed_images = decoder.predict(bottleneck_features)

# Display original, bottleneck, and reconstructed images
plt.figure(figsize=(20, 8))
for i in range(n):
    # Original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(test_images[i])
    plt.title("Original")
    plt.axis("off")

    # Bottleneck embeddings
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(
        bottleneck_features[i].reshape(32, 32), cmap="viridis"
    )  # Adjust shape if needed
    plt.title("Bottleneck")
    plt.axis("off")

    # Reconstructed images
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(reconstructed_images[i])
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()
