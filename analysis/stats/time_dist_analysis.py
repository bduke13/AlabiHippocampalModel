# compare_trials.py
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

DEFAULT_EXPERIMENTS = [
    #"20x20_base",
    "20x20_cross_base",
    "20x20_cross_grid",
    #"20x20_grid"
]

JSON_SUBPATH = "small_medium_large/JSON"

def load_trials(experiment_folder, stats_dir="analysis/stats"):
    json_dir = os.path.join(stats_dir, experiment_folder, JSON_SUBPATH)
    trials = []
    if not os.path.exists(json_dir):
        print(f"Warning: {json_dir} not found.")
        return []
    for fname in sorted(os.listdir(json_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(json_dir, fname), "r") as f:
                data = json.load(f)
            # Extract trial number from filename, sort numerically
            trial_num = int(fname.split("_")[1])
            trials.append((trial_num, data))
    # Sort by trial number
    trials.sort()
    # Only keep data (sorted)
    return [data for (_, data) in trials]

def plot_lines(all_dists, all_times, experiment_names):
    trials = np.arange(1, len(all_dists[0]) + 1)
    plt.figure(figsize=(10,6))
    for dists, label in zip(all_dists, experiment_names):
        plt.plot(trials, dists, marker="o", label=label)
    plt.xlabel("Trial")
    plt.ylabel("Distance Traveled (meters)")
    plt.title("Distance per Trial")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,6))
    for times, label in zip(all_times, experiment_names):
        plt.plot(trials, times, marker="o", label=label)
    plt.xlabel("Trial")
    plt.ylabel("Time Taken (seconds)")
    plt.title("Time per Trial")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bars(all_dists, all_times, experiment_names):
    means_dists = [np.mean(d) for d in all_dists]
    std_dists = [np.std(d) for d in all_dists]

    means_times = [np.mean(t) for t in all_times]
    std_times = [np.std(t) for t in all_times]

    x = np.arange(len(experiment_names))

    plt.figure(figsize=(8,6))
    plt.bar(x, means_dists, yerr=std_dists, capsize=8)
    plt.xticks(x, experiment_names, rotation=15)
    plt.ylabel("Average Distance Traveled (meters)")
    plt.title("Average Distance Traveled Per Experiment\n(with Standard Deviation Error Bars)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,6))
    plt.bar(x, means_times, yerr=std_times, capsize=8)
    plt.xticks(x, experiment_names, rotation=15)
    plt.ylabel("Average Time Taken (seconds)")
    plt.title("Average Time Per Experiment\n(with Standard Deviation Error Bars)")
    plt.tight_layout()
    plt.show()

def main(experiments=None):
    if experiments is None:
        experiments = DEFAULT_EXPERIMENTS
    all_dists = []
    all_times = []
    experiment_names = []

    for exp in experiments:
        trials = load_trials(exp)
        if len(trials) == 0:
            print(f"Skipping {exp} (no data found)")
            continue
        dists = [t.get("total_distance_traveled", np.nan) for t in trials]
        times = [t.get("total_time_secs", np.nan) for t in trials]
        all_dists.append(dists)
        all_times.append(times)
        experiment_names.append(exp)

    if not all_dists:
        print("No valid data found.")
        return

    plot_lines(all_dists, all_times, experiment_names)
    plot_bars(all_dists, all_times, experiment_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare experiment trial stats.")
    parser.add_argument('--experiments', nargs='+', default=DEFAULT_EXPERIMENTS,
                        help="List of experiment folders to compare (default: %(default)s)")
    args = parser.parse_args()
    main(args.experiments)
