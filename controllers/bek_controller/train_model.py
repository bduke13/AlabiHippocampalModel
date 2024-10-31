# %%
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from skimage.transform import resize

# Load the saved images file
images = np.load("recorded_images.npy")
print("Original image shape:", images.shape)

# Use only the first three channels (RGB)
images = images[:, :, :, :3]

# Resize the images to a size that's a multiple of 8
new_size = (96, 192)  # (height, width)


# Function to resize images
def resize_images(images, new_size):
    num_samples = images.shape[0]
    resized_images = np.zeros(
        (num_samples, new_size[0], new_size[1], 3), dtype=np.float32
    )
    for i in range(num_samples):
        resized_images[i] = resize(images[i], new_size, anti_aliasing=True)
    return resized_images


resized_images = resize_images(images, new_size)
print("Resized image shape:", resized_images.shape)

# Convert to float32 type
resized_images = resized_images.astype("float32")

# Preview some images and their data
# plt.figure(figsize=(15, 5))
# for i in range(3):  # Show first 3 images
# plt.subplot(1, 3, i + 1)
# plt.imshow(resized_images[i])
# plt.axis("off")
# plt.title(f"Image {i}")
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


# %%
# Build the autoencoder model with a dense middle layer
def build_autoencoder_with_dense():
    # Input layer
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
    x = layers.Dense(512, activation="relu")(x)  # Bottleneck layer

    # Decoder
    x = layers.Dense(np.prod(shape_before_flattening), activation="relu")(x)
    x = layers.Reshape(shape_before_flattening)(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # (24, 48, 64)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # (48, 96, 32)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # (96, 192, 16)

    # Output layer
    decoded = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    # Create model
    autoencoder = Model(input_img, decoded)
    return autoencoder


# Instantiate and compile the model
autoencoder = build_autoencoder_with_dense()
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

# Train the autoencoder
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
history = autoencoder.fit(
    X_train,
    X_train,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping],
)

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
# Encode and decode some images from the validation set
decoded_imgs = autoencoder.predict(X_val)

# %%
# Display original and reconstructed images
n = 5  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_val[i])
    plt.title("Original")
    plt.axis("off")

    # Reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("Reconstructed")
    plt.axis("off")
plt.show()
