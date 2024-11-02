# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Specify GPUs to use
# Suppress NUMA warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# %%
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import distribute
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

plt.show()

# Split the data into training and validation sets
X_train, X_val = train_test_split(resized_images, test_size=0.1, random_state=42)

# Set batch size for training
BATCH_SIZE = 2


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

    # Flatten and Dense bottleneck
    shape_before_flattening = x.shape[1:]  # Save the shape for later
    x = layers.Flatten()(x)
    bottleneck = layers.Dense(
        1024, activation="relu", activity_regularizer=regularizers.l1(1e-3)
    )(x)

    # Encoder model
    encoder = Model(input_img, bottleneck, name="encoder")

    # Decoder input
    encoded_input = layers.Input(shape=(1024,))
    x = layers.Dense(np.prod(shape_before_flattening), activation="relu")(encoded_input)
    x = layers.Reshape(shape_before_flattening)(x)
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


# Set up the distributed training strategy
strategy = distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

# Create the model within strategy scope
with strategy.scope():
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
    epochs=120,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping],
)

# %%
# Save the encoder and decoder separately
encoder.save("encoder_model.keras")
decoder.save("decoder_model.keras")

# %%
# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.show()

# %%
# Get random indices for visualization
n = 5  # Number of images to display
random_indices = np.random.choice(len(X_val), n, replace=False)

# Get encoded representations and decoded images
encoded_output = encoder.predict(X_val)
decoded_imgs = decoder.predict(encoded_output)

# Create a figure with 3 rows: original, reconstructed, and encoded
plt.figure(figsize=(20, 12))
for i, idx in enumerate(random_indices):
    # Original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_val[idx])
    plt.title("Original")
    plt.axis("off")

    # Reconstructed images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[idx])
    plt.title("Reconstructed")
    plt.axis("off")

    # Encoded representation
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(
        encoded_output[idx].reshape(32, 32), cmap="viridis"
    )  # Reshape to a square for visualization
    plt.title("Encoded")
    plt.axis("off")

plt.tight_layout()
plt.show()
