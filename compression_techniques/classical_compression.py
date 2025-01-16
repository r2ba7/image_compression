import pywt
import pywt.data
from scipy.fftpack import dct, idct
from PIL import Image
import numpy as np
from numpy.linalg import svd

import helpers

class ClassicalCompression:
    def __init__(self, image_handler=None):
        self.image_handler = image_handler
        if self.image_handler.images is None:
            self.image_handler.open_images_to_list()

    # DCT Compression
    def dct_compression(self):
        """Applies DCT-based compression on ImageHandling object's images_array."""
        dct_compressed_images = []
        for img in self.image_handler.images:
            gray_img = helpers.convert_to_grayscale(img)
            img_array = np.array(gray_img)
            dct_array = dct(dct(img_array.T, norm='ortho').T, norm='ortho')
            threshold = 50
            dct_array[np.abs(dct_array) < threshold] = 0
            reconstructed = idct(idct(dct_array.T, norm='ortho').T, norm='ortho')
            dct_compressed_images.append(np.clip(reconstructed, 0, 255).astype(np.uint8))
        
        return dct_compressed_images

    # RLE Compression
    def wavelet_compression(self):
        """Applies Wavelet-based compression on ImageHandling object's images_array."""
        wavelet_compressed_images = []
        # Decomposition using DWT (Discrete Wavelet Transform)
        for img in self.image_handler.images:
            gray_img = helpers.convert_to_grayscale(img)
            img_array = np.array(gray_img)
            
            # Perform the wavelet transform (we'll use 'haar' as a simple wavelet)
            coeffs = pywt.dwt2(img_array, 'haar')
            LL, (LH, HL, HH) = coeffs
            
            # Thresholding: Zero out small coefficients in the high frequency bands
            threshold = 50
            LH[np.abs(LH) < threshold] = 0
            HL[np.abs(HL) < threshold] = 0
            HH[np.abs(HH) < threshold] = 0
            
            # Reconstruct the image from the modified coefficients
            compressed_coeffs = LL, (LH, HL, HH)
            reconstructed = pywt.idwt2(compressed_coeffs, 'haar')
            wavelet_compressed_images.append(np.clip(reconstructed, 0, 255).astype(np.uint8))
        
        return wavelet_compressed_images

    def svd_compression(self, k=40):
        """
        Apply SVD-based compression to each image in the image handler's images array.
        
        Parameters:
        - k: The number of singular values to keep for compression.

        Returns:
        - svd_compressed_images: List of compressed images (using SVD with k components).
        """
        svd_compressed_images = []
        for img in self.image_handler.images:
            gray_img = helpers.convert_to_grayscale(img)
            img_array = np.array(gray_img)

            # Apply SVD
            U, S, V = svd(img_array, full_matrices=False)
            
            # Ensure k is not greater than the smaller dimension
            k = min(k, img_array.shape[0], img_array.shape[1])

            # Create the compressed image using the first k singular values
            compressed_image = np.dot(U[:, :k], np.dot(np.diag(S[:k]), V[:k, :]))

            # Clip the compressed image to [0, 255] and convert to uint8
            svd_compressed_images.append(np.clip(compressed_image, 0, 255).astype(np.uint8))

        return svd_compressed_images
    