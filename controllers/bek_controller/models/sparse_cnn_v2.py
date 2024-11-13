import tensorflow as tf
from tensorflow.keras import layers, Model


def create_models(input_shape):
    # Encoder
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (4, 4), strides=2, activation="relu", padding="same")(
        input_layer
    )  # 48x48x16
    x = layers.Conv2D(32, (4, 4), strides=2, activation="relu", padding="same")(
        x
    )  # 24x24x32
    x = layers.Conv2D(64, (4, 4), strides=2, activation="relu", padding="same")(
        x
    )  # 12x12x64
    x = layers.Flatten()(x)
    latent = layers.Dense(1000, activation="relu", name="latent_space")(x)  # Fc1

    # Create encoder model
    encoder = Model(inputs=input_layer, outputs=latent, name="encoder")

    # Decoder
    latent_input = layers.Input(shape=(1000,))
    x = layers.Dense(12 * 12 * 64, activation="relu")(latent_input)  # Fc2
    x = layers.Reshape((12, 12, 64))(x)  # Reshape back to 12x12x64
    x = layers.Conv2DTranspose(
        64, (4, 4), strides=2, activation="relu", padding="same"
    )(
        x
    )  # 24x24x64
    x = layers.Conv2DTranspose(
        32, (4, 4), strides=2, activation="relu", padding="same"
    )(
        x
    )  # 48x48x32
    x = layers.Conv2DTranspose(
        16, (4, 4), strides=2, activation="relu", padding="same"
    )(
        x
    )  # 96x96x16
    output_layer = layers.Conv2DTranspose(
        3, (3, 3), activation="sigmoid", padding="same"
    )(
        x
    )  # Output 96x96x3

    # Create decoder model
    decoder = Model(inputs=latent_input, outputs=output_layer, name="decoder")

    # Create autoencoder by connecting encoder and decoder
    autoencoder_input = layers.Input(shape=input_shape)
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = Model(inputs=autoencoder_input, outputs=decoded, name="autoencoder")

    # Loss function incorporating the sparsity regularization
    def sparse_loss(y_true, y_pred):
        reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        encoded_output = encoder(y_true)
        z_gram = tf.matmul(
            tf.transpose(encoded_output), encoded_output
        )  # Gramian matrix
        identity_matrix = tf.eye(tf.shape(z_gram)[0])  # Identity matrix
        orthonormal_loss = tf.reduce_mean(tf.square(z_gram - identity_matrix))
        lambda_reg = 0.001  # Hyperparameter controlling sparsity
        total_loss = reconstruction_loss + lambda_reg * orthonormal_loss
        return total_loss

    autoencoder.compile(optimizer="adam", loss=sparse_loss)
    return encoder, decoder, autoencoder


if __name__ == "__main__":
    # Create and compile the models
    input_shape = (96, 96, 3)
    encoder, decoder, autoencoder = create_models(input_shape)

    # Print model summaries
    print("\nEncoder Summary:")
    encoder.summary()
    print("\nDecoder Summary:")
    decoder.summary()
    print("\nAutoencoder Summary:")
    autoencoder.summary()
