# %%
import matplotlib.pyplot as plt

# Generate plots
for i in range(10):
    plt.figure()
    plt.plot([0, 1, 2], [i, i + 1, i + 2])  # Sample plot
    plt.title(f"Plot {i}")
    plt.savefig(f"plot_{i}.png")  # Save each plot

# Create a simple HTML file to view them
with open("plots.html", "w") as f:
    f.write(
        '<html><head><title>Scrollable Plots</title></head><body style="height: 100%; overflow: auto;">'
    )
    for i in range(10):
        f.write(
            f'<img src="plot_{i}.png" style="display: block; margin-bottom: 10px;">'
        )
    f.write("</body></html>")

# Open in default browser
import webbrowser

webbrowser.open("plots.html")
