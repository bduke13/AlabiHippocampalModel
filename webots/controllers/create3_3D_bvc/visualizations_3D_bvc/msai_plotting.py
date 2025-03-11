# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import pandas as pd
import hashlib


if __name__ == "__main__":
    # Set global font to Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"  # For math text

    # Load your CSV
    input_csv = "webots/controllers/create3_3D_bvc/visualizations_3D_bvc/outputs/cosine_similarities.csv"
    df = pd.read_csv(input_csv)
    print(f"Loaded data from: {input_csv}")
    print(df.head())

    # Compute global color range
    global_min = df["cosine_similarity_sum"].min()
    global_max = df["cosine_similarity_sum"].max()
    print(f"Global Cosine Sum Range: [{global_min:.3f}, {global_max:.3f}]")

    # Define name mappings for model numbers
    MODEL_NAMES = {
        # "model0": "Model 0",
        # "model1": "Model 1",
        # "model2": "Model 2",
        # "model3": "Model 3",
        # "model4": "Model 4",
        # --
        "unknown_model": "Model 4",
        # --
    }

    # Define environment mappings based on rotated_X to tilt angles
    ENV_NAMES = {
        # "env1": "Upright",
        # "env2": "30° Tilt",
        # "env3": "45° Tilt",
        # "env4": "60° Tilt",
        # --
        # "env1": "Control",
        # "env2": "Test",
        # --
        "env1": "normal",
        "env2": "no vert",
        "env3": "no dist",
        "env4": "nothing",
    }

    # Desired model/environment order
    MODEL_ORDER = [
        # "model0",
        # "model1",
        # "model2",
        # "model3",
        # "model4",
        # --
        "unknown_model",
        # --
    ]

    ENV_ORDER = [
        # "env1",
        # "env2",
        # "env3",
        # "env4",
        # --
        # "env1",
        # "env2",
        # --
        "env1",
        "env2",
        "env3",
        "env4",
    ]

    n_rows = len(MODEL_ORDER)
    n_cols = len(ENV_ORDER)

    # Create subplots with reduced vertical spacing
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.5 * n_cols, 2.5 * n_rows),
        gridspec_kw={
            "hspace": -0.12,  # Reduced vertical spacing
            "wspace": 0.0,  # Keep horizontal spacing tight
        },
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
                # Normalize sums by total number of data points
                num_points = len(subset)
                normalized_sums = sums / num_points
                sc = ax.scatter(
                    x_vals,
                    y_vals,
                    c=normalized_sums,
                    cmap="viridis",
                    vmin=global_min / num_points,  # Normalize global min/max
                    vmax=global_max / num_points,
                    s=3,  # ~10% smaller than 30
                )
                scatter_obj = sc
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=18)

            # Make each subplot square, remove ticks and spines
            ax.set_aspect("equal", "box")
            ax.set_xticks([])
            ax.set_yticks([])
            # Remove the black bounding box
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Titles/labels with font sizes using mapped names
            if r == 0:
                ax.set_title(ENV_NAMES[env_name], fontsize=15)
            if c == 0:
                ax.set_ylabel(
                    MODEL_NAMES[model_name], rotation=90, labelpad=10, fontsize=15
                )

    # Shrink the plotting area to 88% of figure width, 95% of figure height, with reduced spacing
    plt.tight_layout(rect=[0, 0, 0.88, 0.95], h_pad=0.1, w_pad=-0.1)

    # Make the color bar bigger by increasing fraction from 0.02 -> 0.04
    if scatter_obj is not None:
        cbar = fig.colorbar(
            scatter_obj,
            ax=axes.ravel().tolist(),
            fraction=0.04,  # about twice as large
            pad=0.02,
        )

    plt.show()

    # Create bar plot of sums with square aspect ratio
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle("Mean Spatial Aliasing Index by Environment", fontsize=18)

    # Calculate sums for each model and environment
    sum_data = []
    for env_name in ENV_ORDER:
        env_data = df[df["environment"] == env_name]
        for model_name in MODEL_ORDER:
            model_data = env_data[df["model"] == model_name]
            if len(model_data) > 0:
                # Count the number of data points to get accurate normalization
                num_points = len(model_data)
                # Calculate the total sum and normalize by the square of the number of points
                total_sum = model_data["cosine_similarity_sum"].sum() / (num_points**2)
                sum_data.append(
                    {
                        "Environment": ENV_NAMES[env_name],
                        "Model": MODEL_NAMES[model_name],
                        "Total Sum": total_sum,
                    }
                )

    # Convert to DataFrame for plotting
    sum_df = pd.DataFrame(sum_data)

    # Create grouped bar plot
    sns.barplot(data=sum_df, x="Environment", y="Total Sum", hue="Model", ax=ax)

    # Customize plot
    ax.set_xlabel("", fontsize=18)
    ax.set_ylabel("MSAI", fontsize=18)
    ax.tick_params(labelsize=14)
    ax.legend(title="Model", fontsize=18, title_fontsize=18, loc="lower left")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
