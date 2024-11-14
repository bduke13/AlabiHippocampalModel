# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow import keras
from keras.models import load_model
import webbrowser
import io
import base64
import os

# %%
# Load the encoder model and embeddings
encoder = load_model("encoder_model.keras")
resized_images = np.load("preprocessed_images.npy")
embeddings = encoder.predict(resized_images)

# Load position data
with open("hmap_x.pkl", "rb") as f:
    hmap_x = np.array(pickle.load(f))
with open("hmap_y.pkl", "rb") as f:
    hmap_y = np.array(pickle.load(f))

# Identify dimensions with non-zero activations
non_zero_dims = []
zero_dims = []
for emb_idx in range(embeddings.shape[1]):
    emb_activations = embeddings[:, emb_idx]
    if np.allclose(emb_activations, 0, atol=1e-10):
        zero_dims.append(emb_idx)
    else:
        non_zero_dims.append(emb_idx)

print(f"Total embedding dimensions: {embeddings.shape[1]}")
print(f"Dimensions with non-zero activations: {len(non_zero_dims)}")
print(f"Dimensions with zero activations: {len(zero_dims)}")


def plot_embedding_group(
    embedding_indices,
    x,
    y,
    activations,
    group_index,
):
    """Plot a group of up to 5 embedding dimensions and return as base64 image."""
    # Ensure x, y, and activations have the same length
    min_len = min(len(x), len(y), len(activations))
    x = x[:min_len]
    y = y[:min_len]
    activations = activations[:min_len]

    num_plots = len(embedding_indices)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    fig.suptitle(f"Embedding Dimensions Group {group_index}", fontsize=16)

    if num_plots == 1:
        axes = [axes]

    for ax, emb_idx in zip(axes, embedding_indices):
        # Get activations for this embedding dimension
        emb_activations = activations[:, emb_idx]

        # Create a hexbin plot
        hb = ax.hexbin(
            x,
            y,
            C=emb_activations,
            gridsize=50,
            reduce_C_function=np.mean,
            cmap="viridis",
        )

        plt.colorbar(hb, ax=ax)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title(f"Dimension {emb_idx}")

    plt.tight_layout()

    # Save figure to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return image_base64


def generate_html_report(image_base64_list, zero_dims):
    """Generate an HTML report of all embedding visualizations."""
    html_path = "embeddings_report.html"

    with open(html_path, "w") as f:
        f.write(
            """
        <html>
        <head>
            <title>Embedding Dimension Visualizations</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .group-viz { margin-bottom: 30px; }
                h1 { color: #333; }
            </style>
        </head>
        <body>
            <h1>Embedding Dimension Visualizations</h1>
        """
        )

        num_groups = len(image_base64_list)
        for i in range(num_groups):
            image_base64 = image_base64_list[i]
            f.write(
                f"""
            <div class="group-viz">
                <h3>Group {i}</h3>
                <img src="data:image/png;base64,{image_base64}" style="max-width: 100%;">
            </div>
            """
            )

        if zero_dims:
            f.write("<h2>Embedding Dimensions with Zero Activations</h2>")
            f.write(f"<p>{sorted(zero_dims)}</p>")

        f.write("</body></html>")

    return html_path


# Generate plots for non-zero dimensions, grouped by 5
image_base64_list = []
for group_index, i in enumerate(range(0, len(non_zero_dims), 5)):
    group = non_zero_dims[i : i + 5]
    image_base64 = plot_embedding_group(
        group,
        hmap_x,
        hmap_y,
        embeddings,
        group_index,
    )
    image_base64_list.append(image_base64)

# Generate and open HTML report
try:
    html_path = generate_html_report(image_base64_list, zero_dims)
    # Get absolute path and convert to proper URL format
    abs_path = os.path.abspath(html_path)
    file_url = f'file://{abs_path.replace(os.sep, "/")}'

    print(f"Generated plots for {len(non_zero_dims)} embedding dimensions")
    print(f"HTML report generated at: {abs_path}")
    print(f"Opening report in browser: {file_url}")

    webbrowser.open(file_url)
except Exception as e:
    print(f"Error generating HTML report: {str(e)}")
print("\a")
