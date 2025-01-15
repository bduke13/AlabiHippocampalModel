# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Load your CSV
    input_csv = "controllers/bek_controller/analysis_results/cosine_similarities.csv"
    df = pd.read_csv(input_csv)
    print(f"Loaded data from: {input_csv}")
    print(df.head())

    # Compute global color range
    global_min = df["cosine_similarity_sum"].min()
    global_max = df["cosine_similarity_sum"].max()
    print(f"Global Cosine Sum Range: [{global_min:.3f}, {global_max:.3f}]")

    # Desired model/environment order
    MODEL_ORDER = [
        "2D_250",
        "3D_2L_250_1",
        "3D_2L_250_2",
        "3D_3L_250",
    ]
    ENV_ORDER = [
        "upright",
        "inside_shallow",
        "inside_medium",
        "inside_steep",
    ]

    n_rows = len(MODEL_ORDER)
    n_cols = len(ENV_ORDER)

    # Create subplots with reduced vertical spacing
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.5 * n_cols, 2.5 * n_rows),  # Increased from 2.5 to 3.0
        gridspec_kw={
            "hspace": 0.02,
            "wspace": 0.02,
        },  # Reduced both horizontal and vertical spacing
    )

    # Ensure axes is a 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = np.array([axes]).reshape(n_rows, n_cols)

    scatter_obj = None

    # Plot each (model, environment) subset
    for r, model_name in enumerate(MODEL_ORDER):
        for c, env_name in enumerate(ENV_ORDER):
            ax = axes[r, c]
            subset = df[(df["model"] == model_name) & (df["environment"] == env_name)]

            if len(subset) > 0:
                x_vals = subset["x"].values
                y_vals = subset["y"].values
                sums = subset["cosine_similarity_sum"].values

                # Slightly smaller points
                sc = ax.scatter(
                    x_vals,
                    y_vals,
                    c=sums,
                    cmap="viridis",
                    vmin=global_min,
                    vmax=global_max,
                    s=27,  # ~10% smaller than 30
                )
                scatter_obj = sc
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)

            # Make each subplot square, remove ticks and spines
            ax.set_aspect("equal", "box")
            ax.set_xticks([])
            ax.set_yticks([])
            # Remove the black bounding box
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Titles/labels with font sizes
            if r == 0:
                ax.set_title(env_name, fontsize=12)
            if c == 0:
                ax.set_ylabel(model_name, rotation=90, labelpad=10, fontsize=12)

    # Shrink the plotting area to 88% of figure width, 95% of figure height, with reduced spacing
    plt.tight_layout(
        rect=[0, 0, 0.88, 0.95], h_pad=-0.1, w_pad=-0.1
    )  # Reduced both horizontal and vertical padding

    # Make the color bar bigger by increasing fraction from 0.02 -> 0.04
    if scatter_obj is not None:
        cbar = fig.colorbar(
            scatter_obj,
            ax=axes.ravel().tolist(),
            fraction=0.04,  # about twice as large
            pad=0.02,
        )
        cbar.set_label("Aliasing Index (Global Color Range)", fontsize=14)

    # Move the suptitle a bit higher
    fig.suptitle(
        "Aliasing Index", fontsize=20, x=0.5, y=0.96
    )  # x=0.5 centers the title over the entire figure

    plt.show()
