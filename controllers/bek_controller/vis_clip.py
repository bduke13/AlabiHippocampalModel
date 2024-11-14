# %%

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load your preprocessed images
preprocessed_images = np.load("preprocessed_images.npy")
# Load your encoder model
encoder = load_model("encoder_model.keras")


# Function to preprocess and create CLIP embeddings
def create_clip_embeddings(images):
    embeddings = []
    for img_array in images:
        # Convert the numpy image array to a PIL image and normalize it to [0, 255]
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        inputs = processor(images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )  # Normalize
        embeddings.append(image_features.cpu().numpy().flatten())
    return np.array(embeddings)


# Generate CLIP embeddings for the preprocessed images
clip_embeddings = create_clip_embeddings(preprocessed_images)

# Reduce the dimensionality of the CLIP embeddings to 2D using UMAP
scaler = StandardScaler()
clip_embeddings_scaled = scaler.fit_transform(
    clip_embeddings
)  # Standardize before UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
umap_2d = reducer.fit_transform(clip_embeddings_scaled)

# Get encoder activations (bottleneck features)
bottleneck_features = encoder.predict(preprocessed_images)

# Plot the 2D UMAP representation with encoder activations as an overlay
plt.figure(figsize=(12, 8))
plt.scatter(umap_2d[:, 0], umap_2d[:, 1], c="gray", alpha=0.5, label="CLIP Embeddings")
plt.scatter(
    umap_2d[:, 0],
    umap_2d[:, 1],
    c=np.mean(bottleneck_features, axis=1),
    cmap="viridis",
    label="Encoder Activations",
    edgecolors="k",
)
plt.colorbar(label="Mean Activation (bottleneck features)")
plt.title("2D UMAP Projection of CLIP Embeddings with Encoder Activations")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend()
plt.show()
