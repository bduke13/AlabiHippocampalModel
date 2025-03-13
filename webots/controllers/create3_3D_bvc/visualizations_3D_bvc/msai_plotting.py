# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set global font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"  # For math text

# Experiment folders mapping
EXPERIMENTS = {
    "base_experiment": "Base Experiment",
    "ceilings_experiment": "Ceilings Experiment",
    "different_models_scaling": "Different Models Scaling",
    "obstacles_experiment": "Obstacles Experiment",
    "rotated_walls_experiment": "Rotated Walls Experiment",
}

# Define name mappings for models and environments
MAPPINGS = {
    "base_experiment": {
        "MODEL_NAMES": {
            "base": "2D - 1 Layer",
        },
        "ENV_NAMES": {
            "env1": "Open",
            "env2": "Cross",
        },
        "MODEL_ORDER": ["base"],
        "ENV_ORDER": ["env1", "env2"],
    },
    "ceilings_experiment": {
        "MODEL_NAMES": {
            "3D_bvc": "3D - 4 Layer",
        },
        "ENV_NAMES": {
            "env1": "Control",
            "env2": "Test",
        },
        "MODEL_ORDER": ["3D_bvc"],
        "ENV_ORDER": ["env1", "env2"],
    },
    "different_models_scaling": {
        "MODEL_NAMES": {
            "normal": "Normal",
            "no_vert": "No Vertical Scaling",
            "no_dist": "No Distance Scaling",
            "no_vert_dist": "No Vertical or Distance Scaling",
        },
        "ENV_NAMES": {
            "env1": "Standard",
            "env2": "No Vertical",
            "env3": "No Distance",
            "env4": "None",
        },
        "MODEL_ORDER": ["normal", "no_vert", "no_dist", "no_vert_dist"],
        "ENV_ORDER": ["env1", "env2", "env3", "env4"],
    },
    "obstacles_experiment": {
        "MODEL_NAMES": {
            "3D_bvc": "3D Boundary Vector Cells",
        },
        "ENV_NAMES": {
            "env1": "Same Obstacles",
            "env2": "Different Obstacles",
        },
        "MODEL_ORDER": ["3D_bvc"],
        "ENV_ORDER": ["env1", "env2"],
    },
    "rotated_walls_experiment": {
        "MODEL_NAMES": {
            "model0": "2D - 1 Layer",
            "model1": "3D - 1 Layer",
            "model2": "3D - 2 Layer",
            "model3": "3D - 3 Layer",
            "model4": "3D - 4 Layer",
        },
        "ENV_NAMES": {
            "env1": "Upright",
            "env2": "30° Tilt",
            "env3": "45° Tilt",
            "env4": "60° Tilt",
        },
        "MODEL_ORDER": ["model0", "model1", "model2", "model3", "model4"],
        "ENV_ORDER": ["env1", "env2", "env3", "env4"],
    },
}

# Root directory for CSV files
OUTPUT_DIR = "webots/controllers/create3_3D_bvc/visualizations_3D_bvc/outputs"

# Allow user to select an experiment
# print("Available experiments:")
# for key, value in EXPERIMENTS.items():
#    print(f"  {key}: {value}")
# selected_experiment = input("Enter the experiment folder name: ")

selected_experiment = "base_experiment"

if selected_experiment not in EXPERIMENTS:
    raise ValueError("Invalid experiment selection!")

# Load appropriate mappings
mapping = MAPPINGS[selected_experiment]
MODEL_NAMES = mapping["MODEL_NAMES"]
ENV_NAMES = mapping["ENV_NAMES"]
MODEL_ORDER = mapping["MODEL_ORDER"]
ENV_ORDER = mapping["ENV_ORDER"]

# Load data
csv_path = os.path.join(OUTPUT_DIR, selected_experiment, "cosine_similarities.csv")
df = pd.read_csv(csv_path)
print(f"Loaded data from: {csv_path}")

# Replace NaN values with 0.0 in the entire dataframe
df = df.fillna(0.0)
print("Replaced any NaN values with 0.0")

# Compute global color range
global_min = df["cosine_similarity_sum"].min()
global_max = df["cosine_similarity_sum"].max()

# Adjust figure size for better layout
fig_size = (4 * len(ENV_ORDER), 4 * len(MODEL_ORDER))
point_size = 10

fig, axes = plt.subplots(len(MODEL_ORDER), len(ENV_ORDER), figsize=fig_size)
scatter_obj = None

for r, model_name in enumerate(MODEL_ORDER):
    for c, env_name in enumerate(ENV_ORDER):
        ax = axes[r, c] if len(MODEL_ORDER) > 1 else axes[c]
        subset = df[(df["model"] == model_name) & (df["environment"] == env_name)]

        if not subset.empty:
            sc = ax.scatter(
                subset["x"],
                subset["y"],
                c=subset["cosine_similarity_sum"],
                cmap="viridis",
                vmin=global_min,
                vmax=global_max,
                s=point_size,
            )
            scatter_obj = sc
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", "box")
        for spine in ax.spines.values():
            spine.set_visible(False)

        if r == 0:
            ax.set_title(ENV_NAMES.get(env_name, env_name), fontsize=12)
        if c == 0:
            ax.set_ylabel(MODEL_NAMES.get(model_name, model_name), fontsize=12)

plt.subplots_adjust(left=0.05, right=0.85, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
fig.suptitle(
    f"Spatial Aliasing Heatmaps: {EXPERIMENTS[selected_experiment]}",
    fontsize=14,
    y=0.98,
)
if scatter_obj:
    cbar_ax = fig.add_axes([0.87, 0.3, 0.03, 0.5])  # Adjust colorbar position
    plt.colorbar(scatter_obj, cax=cbar_ax)

plt.show()

# Create MSAI bar plot
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Mean Spatial Aliasing Index by Environment", fontsize=12, y=0.94)

sum_data = []
for env_name in ENV_ORDER:
    for model_name in MODEL_ORDER:
        model_data = df[(df["model"] == model_name) & (df["environment"] == env_name)]
        if not model_data.empty:
            total_sum = model_data["cosine_similarity_sum"].mean()
            sum_data.append(
                {
                    "Environment": ENV_NAMES.get(env_name, env_name),
                    "Model": MODEL_NAMES.get(model_name, model_name),
                    "Total Sum": total_sum,
                }
            )

sum_df = pd.DataFrame(sum_data)
sns.barplot(data=sum_df, x="Environment", y="Total Sum", hue="Model", ax=ax)
ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.6)
ax.set_ylabel("MSAI", fontsize=12)
ax.legend(title="Model", fontsize=12, title_fontsize=16, loc="best")
plt.tight_layout()
plt.show()

# Print the MSAI metrics to the terminal
print("\nMean Spatial Aliasing Index (MSAI) Metrics:")
print(sum_df.to_string(index=False))

# Print summary statistics grouped by environment and model
print("\nMSAI Summary Statistics:")
msai_summary = (
    sum_df.groupby(["Environment", "Model"])["Total Sum"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
)
print(msai_summary.to_string(index=False))
