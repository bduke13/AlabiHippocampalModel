import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import layers, models
from keras import Model


def create_models():
    # Input layer for encoder
    input_img = layers.Input(shape=(100, 200, 3))

    # Encoder
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # (50, 100, 16)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # (25, 50, 32)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # (13, 25, 64)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # (7, 13, 128)

    # Flatten and Dense bottleneck
    shape_before_flattening = x.shape[1:]  # Save the shape for later
    x = layers.Flatten()(x)
    bottleneck = layers.Dense(
        200,
        activation="relu",  # activity_regularizer=regularizers.l1(1e-5)
    )(x)

    # Encoder model
    encoder = Model(input_img, bottleneck, name="encoder")

    # Decoder input
    encoded_input = layers.Input(shape=(200,))
    x = layers.Dense(np.prod(shape_before_flattening), activation="relu")(encoded_input)
    x = layers.Reshape(shape_before_flattening)(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # (14, 26, 128)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # (28, 52, 64)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # (56, 104, 32)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)  # (112, 208, 16)

    # Add a cropping layer to ensure the final size matches 100x200
    x = layers.Cropping2D(((6, 6), (4, 4)))(x)  # Crop to (100, 200)

    # Output layer for decoder
    decoded_output = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    # Decoder model
    decoder = Model(encoded_input, decoded_output, name="decoder")

    # Full autoencoder model that connects encoder and decoder
    autoencoder = Model(input_img, decoder(encoder(input_img)))

    return encoder, decoder, autoencoder


if __name__ == "__main__":
    # Create the model
    encoder, decoder, autoencoder = build_separate_encoder_decoder()
    autoencoder.compile(optimizer="adam", loss="mse")

    # Print model summaries
    print("\nEncoder Summary:")
    encoder.summary()
    print("\nDecoder Summary:")
    decoder.summary()
    print("\nAutoencoder Summary:")
    autoencoder.summary()
