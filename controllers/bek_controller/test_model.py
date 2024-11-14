# %%
# autoencoder_test.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model

# %%
# Load the saved encoder and decoder models
encoder = load_model("encoder_model.keras")
decoder = load_model("decoder_model.keras")

# Load preprocessed images
preprocessed_images = np.load("preprocessed_images.npy")

n = 10  # Number of images to display
random_indices = np.random.choice(len(preprocessed_images), n, replace=False)

# Get encoded representations and decoded images for the selected samples
selected_samples = preprocessed_images[random_indices]
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
