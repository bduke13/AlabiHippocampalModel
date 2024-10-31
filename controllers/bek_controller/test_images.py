# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# Load the saved images file
images = np.load("recorded_images.npy")

# %%
# Display each channel separately
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
channels = ["Red", "Green", "Blue", "Alpha"]

for idx, (ax, channel) in enumerate(zip(axes.flat, channels)):
    ax.imshow(images[2000][:, :, idx], cmap="gray")
    ax.set_title(f"{channel} Channel")
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
print(images[0].shape)
# (360, 720, 4)
