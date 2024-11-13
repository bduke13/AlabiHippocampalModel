# %%
import numpy as np
from skimage.transform import resize


def resize_images(images, new_size):
    """Resize a batch of images to the specified size."""
    num_samples = images.shape[0]
    resized_images = np.zeros(
        (num_samples, new_size[0], new_size[1], 3), dtype=np.float32
    )
    for i in range(num_samples):
        resized_images[i] = resize(images[i], new_size, anti_aliasing=True)
    return resized_images


def preprocess_images(input_file, output_file, new_size):
    # Load the saved images file
    print(f"Loading images from {input_file}")
    images = np.load(input_file)
    print("Original image shape:", images.shape)

    # Use only the first three channels (RGB)
    images = images[:, :, :, :3]

    print("Resizing images...")
    resized_images = resize_images(images, new_size)
    print("Resized image shape:", resized_images.shape)

    # Convert to float32 type
    resized_images = resized_images.astype("float32")

    # Save the preprocessed images
    print(f"Saving preprocessed images to {output_file}")
    np.save(output_file, resized_images)
    print("Done!")


# Default filenames
INPUT_FILE = "recorded_images.npy"
OUTPUT_FILE = "preprocessed_images.npy"
# Resize the images to a size that's a multiple of 8
new_size = (96, 96)  # (height, width)
preprocess_images(INPUT_FILE, OUTPUT_FILE, new_size)
