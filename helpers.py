import os
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def pick_random_images_from_path(base_path, num_samples=None):
    selected_images_paths = []
    if not os.path.exists(base_path):
        raise ValueError(f"The path {base_path} does not exist.")
    
    all_images = []
    for dir in os.listdir(base_path):
        class_path = os.path.join(base_path, dir)
        if os.path.isdir(class_path):  # If it's a directory
            image_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            all_images.extend(image_files)
        else:  # If it's not a directory, check for images in base_path
            if os.path.isfile(class_path):
                all_images.append(class_path)
    
    if num_samples is not None:
        if len(all_images) < num_samples:
            raise ValueError(f"Not enough images in total to select {num_samples} samples.")
        
        selected_images_paths = random.sample(all_images, num_samples)
        return selected_images_paths
    else:
        return all_images

def preview_images(images, num_samples=None):
    """
    Preview a set of images in a grid. Optionally take a subset of the images to preview.
    
    Parameters:
    - images: List of images to display.
    - num_samples: Number of images to sample and display. If None, all images are shown.
    """
    if num_samples is not None:
        # Limit the number of images to display
        images = images[:num_samples]  # Take the first `num_samples` images
    
    num_images = len(images)
    rows = int(num_images ** 0.5)  # Approx square root to determine rows
    cols = (num_images + rows - 1) // rows  # Ensure enough columns to fit images
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()  # Flatten the axes array for easier iteration

    for i, img in enumerate(images):
        ax = axes[i]
        
        # Check if the image is 3D (color) or 2D (grayscale)
        if len(img.shape) == 3:  # 3D image (e.g., RGB)
            ax.imshow(img)
        elif len(img.shape) == 2:  # 2D image (grayscale)
            ax.imshow(img, cmap="gray")
        else:
            print(f"Image {i+1} has an unsupported shape: {img.shape}")
            ax.axis('off')  # Disable the axis for unsupported images
            continue
        
        ax.axis('off')  # Hide axes
        ax.set_title(f"Image {i+1}")

    # Hide any extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')  
    
    plt.tight_layout()
    plt.show()


def convert_images_to_arrays(images):
    images_array = np.array([np.array(img) for img in images])
    return images_array

# Helper method to ensure images are in grayscale
def convert_to_grayscale(img):
    """
    Converts a PIL image to grayscale using the 'L' mode.
    """
    if isinstance(img, list):
        return [convert_to_grayscale(i) for i in img]

    return img.convert("L")

def save_images(images, path):
    """Saves the images to the specified path in .jpg format."""
    for i, img in enumerate(images):
        # Ensure img is a PIL Image object
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)  # Convert to PIL Image if it's a numpy array

        img.save(f"{path}/image_{i+1}.jpg", "JPEG")
    print(f"Images saved to {path}")

def get_path_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

def convert_grayscale_to_rgb(images):
    """
    Convert grayscale images (1 channel) to RGB (3 channels) by duplicating the channel.
    
    Args:
        images (tf.Tensor): Input grayscale images of shape (batch_size, height, width, 1).
    
    Returns:
        tf.Tensor: RGB images of shape (batch_size, height, width, 3).
    """
    return tf.image.grayscale_to_rgb(images)