# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
import matplotlib.colors as mcolors
import webbrowser


def plot_boundary_cells_group(
    cell_indices,
    hmap_x,
    hmap_y,
    hmap_to_plot,
    colors_rgb,
    group_index,
    names_file="cell_names.txt",  # Add the file parameter
    output_dir="visualizations/outputs/",
    save_plot=True,
    show_plot=False,
    return_plot=False,
):
    """
    Plots a hexbin plot for a group of place cells and saves or shows it.
    """
    # Load cell names from the file
    with open(names_file, "r") as f:
        cell_names = [line.strip() for line in f]

    # Create figure with 5 subplots side by side
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"Boundary Cells Group {group_index}", fontsize=16)

    for idx, (ax, cell_index) in enumerate(zip(axes, cell_indices)):
        # Get activations for this cell
        activations = hmap_to_plot[:, cell_index]

        # Cell name from the loaded file
        name = (
            cell_names[cell_index]
            if cell_index < len(cell_names)
            else f"Cell {cell_index}"
        )

        # Color for this cell
        color_rgb = colors_rgb[
            cell_index % len(colors_rgb)
        ]  # Use modulo to avoid index out of range

        # Create a hexbin plot
        hb = ax.hexbin(
            hmap_x,
            hmap_y,
            C=activations,
            gridsize=50,
            reduce_C_function=np.mean,
            cmap=None,
            edgecolors="none",
        )

        # Get aggregated activations per bin
        counts = hb.get_array()

        # Normalize counts for alpha values
        max_count = counts.max()
        if max_count > 0:
            counts_normalized = counts / max_count
        else:
            counts_normalized = counts

        # Create RGBA colors
        rgba_colors = np.zeros((len(counts), 4))
        rgba_colors[:, 0:3] = color_rgb
        rgba_colors[:, 3] = counts_normalized

        # Set the facecolors of the hexbin collection
        hb.set_facecolors(rgba_colors)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(name)  # Use the name from the file

    plt.tight_layout()

    # Save the plot if save_plot flag is True
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"boundary_cells_group_{group_index}.jpg")
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        print(f"Saved plot for boundary cells group {group_index} to {file_path}")

    # Show the plot if show_plot flag is True
    if show_plot:
        plt.show()

    # Return the figure if return_plot flag is True
    if return_plot:
        return fig

    # Close the figure to save memory
    plt.close(fig)


def generate_html_report(cell_indices, output_dir):
    """Generate an HTML report of all boundary cell visualizations."""
    html_path = os.path.join(output_dir, "boundary_cells_report.html")

    with open(html_path, "w") as f:
        f.write(
            """
        <html>
        <head>
            <title>Boundary Cell Visualizations</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .group-viz { margin-bottom: 30px; }
                h1 { color: #333; }
            </style>
        </head>
        <body>
            <h1>Boundary Cell Visualizations</h1>
        """
        )

        num_groups = (len(cell_indices) + 4) // 5  # Round up division by 5
        for i in range(num_groups):
            f.write(
                f"""
            <div class="group-viz">
                <h3>Group {i}</h3>
                <img src="boundary_cells_group_{i}.jpg" style="max-width: 100%;">
            </div>
            """
            )

        f.write("</body></html>")

    return html_path


if __name__ == "__main__":
    # Create the output directory
    output_dir = "visualizations/outputs/"
    os.makedirs(output_dir, exist_ok=True)

    # Load the colors list
    with open("visualizations/colors.json", "r") as f:
        colors = json.load(f)

    # Convert hex colors to RGB format
    colors_rgb = [mcolors.to_rgb(c) for c in colors]

    # Load hmap data
    with open("hmap_x.pkl", "rb") as f:
        hmap_x = np.array(pickle.load(f))
    with open("hmap_y.pkl", "rb") as f:
        hmap_y = np.array(pickle.load(f))
    with open("hmap_bvc.pkl", "rb") as f:
        hmap_to_plot = np.asarray(pickle.load(f))

    # Use sequential indices without sorting
    cell_indices = np.arange(hmap_to_plot.shape[1])

    # Generate and save plots in groups of 5
    for i in range(0, len(cell_indices), 5):
        group = cell_indices[i : i + 5]
        # Pad the group with None if it's not complete
        while len(group) < 5:
            group = np.append(group, group[-1])

        group_index = i // 5
        plot_boundary_cells_group(
            group,
            hmap_x,
            hmap_y,
            hmap_to_plot,
            colors_rgb,
            group_index,
            output_dir=output_dir,
            save_plot=True,
            show_plot=False,
        )

    # Generate HTML report
    html_path = generate_html_report(cell_indices, output_dir)

    # Open in default browser
    webbrowser.open(f"file://{os.path.abspath(html_path)}")

    print(f"Processed plots for {len(cell_indices)} cell(s).")
    print(f"HTML report generated at: {html_path}")
