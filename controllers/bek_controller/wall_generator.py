import random

def generate_random_wall(enclosure_size):
    # Wall dimensions
    wall_length = 1.0
    wall_height = 1.2
    wall_thickness = 0.15

    # Generate random position
    # Ensure the wall doesn't exceed the enclosure bounds
    max_translation = (enclosure_size / 2) - (wall_length / 2)
    x_translation = random.uniform(-max_translation, max_translation)
    z_translation = random.uniform(-max_translation, max_translation)

    # Generate random orientation (rotation about the Y-axis)
    rotation_angle = random.uniform(0, 2 * 3.14159)  # Radians

    # Wall data in VRML format
    wall_vrml = f"""
    Wall {{
      translation {x_translation:.2f} 0 {z_translation:.2f}
      rotation 0 1 0 {rotation_angle:.4f}
      name "wall{rotation_angle:.4f}"
      size {wall_length} {wall_height} {wall_thickness}
      appearance Roughcast {{
        colorOverride 0.8 0.45 0.25
        textureTransform TextureTransform {{
          scale 1 1.2
        }}
      }}
    }}
    """

    return wall_vrml

# Example usage
enclosure_size = 5.0  # Length of the square enclosure side
random_wall = generate_random_wall(enclosure_size)
print(random_wall)
