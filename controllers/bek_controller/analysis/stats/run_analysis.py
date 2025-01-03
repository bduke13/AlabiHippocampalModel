import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Define the directory containing the JSON files
directory = r"controllers\bek_controller\\analysis\stats\\multiscale\world1"

# Load JSON files into a DataFrame
def load_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as f:
                data.append(json.load(f))
    return pd.DataFrame(data)

# Load the data
df = load_json_files(directory+"\\JSON")

# Convert start_location to a string for grouping
df['corner'] = df['start_location'].apply(lambda x: str(x))

# Compute averages per corner
avg_df = df.groupby('corner').mean()

# Create directory for saving images if it doesn't exist
output_dir = directory
os.makedirs(output_dir, exist_ok=True)

# Plot and save the results
metrics = ['total_distance_traveled', 'total_time_secs', 'turn_count', 'collision_count']
for metric in metrics:
    plt.figure()
    avg_df[metric].plot(kind='bar', title=f"Average {metric} per Corner", ylabel=metric, xlabel='Corner')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "plots",f"average_{metric}_per_corner.png")
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory
    print(f"Saved {metric} plot to {save_path}")

# Print the averages for reference
print(avg_df)
