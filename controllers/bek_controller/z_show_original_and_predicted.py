# %%

import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import webbrowser

# Load the saved encoder and decoder models
encoder = load_model("encoder_model.keras")
decoder = load_model("decoder_model.keras")

# Load preprocessed images
preprocessed_images = np.load("preprocessed_images.npy")

# Select random images for visualization
n_images = 100
random_indices = np.random.choice(len(preprocessed_images), n_images, replace=False)
random_indices.sort()  # Sort indices for organized display
test_images = preprocessed_images[random_indices]

# Generate reconstructions
bottleneck_features = encoder.predict(test_images)
reconstructed_images = decoder.predict(bottleneck_features)

# Create and save comparison plots
for i in range(n_images):
    plt.figure(figsize=(12, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(test_images[i])
    plt.title("Original Image")
    plt.axis("off")

    # Reconstructed image
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_images[i])
    plt.title("Reconstructed Image")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"models/outputs/comparison_{i}.png")
    plt.close()

# Create HTML file
with open("reconstructions.html", "w") as f:
    f.write(
        """
    <html>
    <head>
        <title>Image Reconstructions</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .comparison { margin-bottom: 30px; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <h1>Original vs Reconstructed Images</h1>
    """
    )

    for i in range(n_images):
        f.write(
            f"""
        <div class="comparison">
            <h3>Image {i+1} (Original Index: {random_indices[i]})</h3>
            <img src="models/outputs/comparison_{i}.png" style="max-width: 100%;">
        </div>
        """
        )

    f.write("</body></html>")

# Open in default browser
webbrowser.open("reconstructions.html")
