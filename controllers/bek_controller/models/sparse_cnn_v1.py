import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras import Model


# Build separate encoder and decoder models
def create_models():
    # Input layer for encoder
    input_img = layers.Input(shape=(96, 96, 3))

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
        200,
        activation="relu",  # activity_regularizer=regularizers.l1(1e-5)
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
