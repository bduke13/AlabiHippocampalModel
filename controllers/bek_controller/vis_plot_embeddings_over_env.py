# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model
import os
import webbrowser

# Load the encoder model and embeddings
encoder = load_model("encoder_model.keras")
resized_images = np.load("preprocessed_images.npy")
embeddings = encoder.predict(resized_images)

# Load position data
with open("hmap_x.pkl", "rb") as f:
    hmap_x = np.array(pickle.load(f))
with open("hmap_y.pkl", "rb") as f:
    hmap_y = np.array(pickle.load(f))

# Create output directory
output_dir = "embedding_activation_plots/"
os.makedirs(output_dir, exist_ok=True)


def plot_embedding_group(
    embedding_indices,
    x,
    y,
    activations,
    group_index,
    output_dir="embedding_activation_plots/",
    save_plot=True,
    show_plot=False,
):
    # Ensure x, y and activations have the same length
    min_len = min(len(x), len(y), len(activations))
    x = x[:min_len]
    y = y[:min_len]
    activations = activations[:min_len]
    """Plot a group of 5 embedding dimensions using hexbin."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"Embedding Dimensions Group {group_index}", fontsize=16)

    for idx, (ax, emb_idx) in enumerate(zip(axes, embedding_indices)):
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

    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"embedding_group_{group_index}.jpg")
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        print(f"Saved plot for embedding group {group_index} to {file_path}")

    if show_plot:
        plt.show()

    plt.close(fig)


def generate_html_report(num_embeddings, output_dir):
    """Generate an HTML report of all embedding visualizations."""
    html_path = os.path.join(output_dir, "embeddings_report.html")

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

        num_groups = (num_embeddings + 4) // 5  # Round up division by 5
        for i in range(num_groups):
            f.write(
                f"""
            <div class="group-viz">
                <h3>Group {i}</h3>
                <img src="embedding_group_{i}.jpg" style="max-width: 100%;">
            </div>
            """
            )

        f.write("</body></html>")

    return html_path


# Generate plots in groups of 5
num_embeddings = embeddings.shape[1]
for i in range(0, num_embeddings, 5):
    group = list(range(i, min(i + 5, num_embeddings)))
    # Pad the group with the last index if it's not complete
    while len(group) < 5:
        group.append(group[-1])

    group_index = i // 5
    plot_embedding_group(
        group,
        hmap_x,
        hmap_y,
        embeddings,
        group_index,
        output_dir=output_dir,
        save_plot=True,
        show_plot=False,
    )

# Generate and open HTML report
try:
    html_path = generate_html_report(num_embeddings, output_dir)
    webbrowser.open(f"file://{os.path.abspath(html_path)}")
    print(f"Generated plots for {num_embeddings} embedding dimensions")
    print(f"HTML report generated at: {html_path}")
except Exception as e:
    print(f"Error generating HTML report: {str(e)}")

print(f"Generated plots for {num_embeddings} embedding dimensions")
print(f"HTML report generated at: {html_path}")
