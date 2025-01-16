from PIL import Image
import numpy as np

class ImageHandling():
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self._images = None

    def open_images_to_list(self, target_size=(200, 200)):
        if isinstance(self.image_paths, str):  # Single path input
            self.image_paths = [self.image_paths]  # Convert to list for uniform processing
        
        self._images = []
        for path in self.image_paths:
            try:
                img = Image.open(path)
                img_resized = img.resize(target_size)
                self._images.append(img_resized)
            except Exception as e:
                print(f"Error opening image {path}: {e}")

    @property
    def images(self):
        if self._images is None:
            raise ValueError("Images have not been loaded. Use open_images_to_list() method first.")
            
        return self._images