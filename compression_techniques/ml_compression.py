from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans
import numpy as np

import helpers

class MLCompression:
    def __init__(self, train_image_handler, test_image_handler):
        self.train_image_handler = train_image_handler
        self.test_image_handler = test_image_handler
        if self.train_image_handler.images is None:
            self.train_image_handler.open_images_to_list()

        if self.test_image_handler.images is None:
            self.test_image_handler.open_images_to_list()

        self.pca_model = None
        self.kmeans_model = None

    def train_pca(self, k):
        """Trains PCA model on training images."""
        pca_data = []
        for img in self.train_image_handler.images:
            gray_img = helpers.convert_to_grayscale(img)
            img_array = np.array(gray_img)
            img_array = img_array.reshape(-1)  # Flatten the image for PCA
            pca_data.append(img_array)
        
        pca_data = np.array(pca_data)
        self.pca_model = IncrementalPCA(n_components=k)
        self.pca_model.fit(pca_data)

    def train_kmeans(self, n_clusters):
        """Trains K-means model on training images."""
        kmeans_data = []
        for img in self.train_image_handler.images:
            gray_img = helpers.convert_to_grayscale(img)
            img_array = np.array(gray_img)
            img_array = img_array.reshape(-1)
            kmeans_data.append(img_array)
        
        kmeans_data = np.array(kmeans_data).reshape(-1, 1)
        self.kmeans_model = KMeans(n_clusters=n_clusters)
        self.kmeans_model.fit(kmeans_data)

    def pca_compress(self, img):
        """Compresses the image using the trained PCA model."""
        gray_img = helpers.convert_to_grayscale(img)  # Convert image to grayscale (PIL.Image)
        img_array = np.array(gray_img)  # Convert the grayscale image to a NumPy array
        img_shape = img_array.shape  # Get the shape of the NumPy array
        img_array_flattened = img_array.reshape(-1)  # Flatten the image for PCA
        compressed = self.pca_model.transform(img_array_flattened.reshape(1, -1))  # Apply PCA compression
        reconstructed = self.pca_model.inverse_transform(compressed)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)  # Reconstruct the image
        return reconstructed.reshape(img_shape)

    def kmeans_compress(self, img):
        """Compresses the image using the trained K-means model."""
        gray_img = helpers.convert_to_grayscale(img)  # Convert image to grayscale (PIL.Image)
        img_array = np.array(gray_img)  # Convert the grayscale image to a NumPy array
        img_shape = img_array.shape  # Get the shape of the NumPy array
        img_array_flattened = img_array.reshape(-1, 1)  # Flatten to 1D array for K-means
        compressed = self.kmeans_model.predict(img_array_flattened)  # Apply K-means compression
        centers = self.kmeans_model.cluster_centers_.reshape(-1)  # Get centroids
        decompressed = centers[compressed].reshape(img_shape)
        decompressed = np.clip(decompressed, 0, 255).astype(np.uint8)
        return decompressed

    def compress_images(self, compression_type="pca"):
        """Compress all images in the test set using the trained model."""
        compressed_images = []
        for img in self.test_image_handler.images:
            if compression_type == "pca":
                compressed_images.append(self.pca_compress(img))
            elif compression_type == "kmeans":
                compressed_images.append(self.kmeans_compress(img))
            else:
                raise ValueError("Unsupported compression type")
        
        return compressed_images
