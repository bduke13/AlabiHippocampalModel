import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import torch

# Globally load the EfficientNetV2B0 model with ImageNet weights,
# excluding the top classification layer, and use global average pooling to output a 1280-dimensional feature vector.
model = EfficientNetV2B0(include_top=False, weights="imagenet", pooling="avg")
# Build a model that outputs the penultimate layer (before global pooling)
intermediate_model = Model(inputs=model.input, outputs=model.layers[-2].output)
# Set the model to inference mode (disable training behaviors)
model.trainable = False

def get_visual_output(rgb_image):
    """
    Extracts a 1280-dimensional visual feature vector from the input RGB image using
    TensorFlow Keras's EfficientNetV2B0 model.

    Parameters:
      rgb_image: numpy.ndarray, shape (H, W, 3), RGB image data with pixel values in the range 0-255.

    Returns:
      feature: torch.Tensor, shape (1280,), dtype=float32
    """
    # Convert the numpy array to a PIL Image and resize it to (224, 224)
    img = Image.fromarray(rgb_image.astype('uint8'))
    img = img.resize((224, 224))
    # Convert the PIL Image back to a numpy array and add a batch dimension
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

    # Apply EfficientNetV2 preprocessing (normalization, channel reordering, etc.)
    img_array = preprocess_input(img_array)

    # Use the model to predict features
    # Extract features from the second-to-last layer
    feature_map = intermediate_model.predict(img_array)  # (1, H, W, C)
    pooled = np.mean(feature_map, axis=(1, 2))  # (1, C)
    features = pooled.flatten()  # (C,) == (1280,)

    # Convert the numpy array to a torch.Tensor and return it
    return torch.tensor(features, dtype=torch.float)