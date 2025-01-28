# plot_placecell_metrics.py

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
bek_controller_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, bek_controller_dir)

def main():
    # 1) Load weight_change_history
    weight_change_path = os.path.join(bek_controller_dir, "weight_change_history.pkl")
    metrics_over_time_path = os.path.join(bek_controller_dir, "place_cell_metrics_over_time.pkl")

    try:
        with open(weight_change_path, "rb") as f:
            weight_change_history = pickle.load(f)
    except FileNotFoundError:
        weight_change_history = None
        print("No weight_change_history found.")

    try:
        with open(metrics_over_time_path, "rb") as f:
            mc_data = pickle.load(f)
            partial_metrics_list = mc_data["partial_metrics"]
            final_metrics = mc_data["final_metrics"]
    except FileNotFoundError:
        partial_metrics_list = []
        final_metrics = None
        print("No place_cell_metrics_over_time found.")

    # 2) Plot weight change if available
    if weight_change_history and len(weight_change_history) > 0:
        plt.figure()
        plt.plot(weight_change_history, label="Oja's Weight Change")
        plt.xlabel("Time Steps")
        plt.ylabel("||ΔW_in||")
        plt.title("Weight Change Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    # 3) Plot partial metrics if we have them
    # partial_metrics_list = [ {'step': s, 'max_firing': [...], 'std_size': [...], 'skewness': [...] }, ...]
    if len(partial_metrics_list) > 0:
        steps = [pm['step'] for pm in partial_metrics_list]

        # We'll compute average (or median) across cells for each partial metric to see how they evolve
        avg_max_firing = []
        avg_std_size = []
        avg_skew = []

        for pm in partial_metrics_list:
            mf = np.array(pm['max_firing'])
            ss = np.array(pm['std_size'])
            sk = np.array(pm['skewness'])

            avg_max_firing.append(np.mean(mf))
            avg_std_size.append(np.mean(ss))
            avg_skew.append(np.mean(sk))

        # (A) Average Max Firing Over Time
        plt.figure(figsize=(8, 6))
        plt.plot(steps, avg_max_firing, marker='o', label="Average Max Firing")
        plt.xlabel("Steps")
        plt.ylabel("Firing Rate")
        plt.title("Average Max Firing Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

        # (B) Average Field Size Over Time
        plt.figure(figsize=(8, 6))
        plt.plot(steps, avg_std_size, marker='o', color='green', label="Average STD Field Size")
        plt.xlabel("Steps")
        plt.ylabel("Field STD (units)")
        plt.title("Average Place Field Size Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

        # (C) Average Skewness Over Time
        plt.figure(figsize=(8, 6))
        plt.plot(steps, avg_skew, marker='o', color='red', label="Average Skewness")
        plt.xlabel("Steps")
        plt.ylabel("Skewness")
        plt.title("Average Skewness Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        print("No partial_metrics to plot or partial_metrics_list empty.")

    # 4) Optionally, final_metrics if you want a histogram
    if final_metrics:
        # final_metrics = {"max_firing": [...], "std_size": [...], "skewness": [...]}
        plt.figure(figsize=(8,6))
        plt.hist(final_metrics['max_firing'], bins=30, alpha=0.7, color='blue')
        plt.xlabel("Max Firing")
        plt.ylabel("Cell Count")
        plt.title("Histogram of Max Firing (Final)")
        plt.show()

        plt.figure(figsize=(8,6))
        plt.hist(final_metrics['std_size'], bins=30, alpha=0.7, color='green')
        plt.xlabel("STD Field Size")
        plt.ylabel("Cell Count")
        plt.title("Histogram of Field Size (Final)")
        plt.show()

        plt.figure(figsize=(8,6))
        plt.hist(final_metrics['skewness'], bins=30, alpha=0.7, color='red')
        plt.xlabel("Skewness")
        plt.ylabel("Cell Count")
        plt.title("Histogram of Skewness (Final)")
        plt.show()

if __name__ == "__main__":
    main()
