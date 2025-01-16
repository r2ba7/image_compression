from math import log10, sqrt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio

class Metrics:
    def __init__(self, original_images=None, compressed_images=None):
        self.original_images = original_images
        self.compressed_images = compressed_images

    @staticmethod
    def calculate_psnr(original, compressed, max_pixel=255.0):
        """Calculate Peak Signal-to-Noise Ratio (PSNR)."""
        if original.shape != compressed.shape:
            raise ValueError("Input images must have the same dimensions.")
        
        mse = np.mean((original - compressed) ** 2)
        if mse == 0: return 100
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr

    @staticmethod
    def calculate_ssim(original, compressed):
        """Calculate Structural Similarity Index (SSIM)."""
        if original.shape != compressed.shape:
            raise ValueError("Input images must have the same dimensions.")
        
        # Check the dtype of the images to set the correct data_range
        if original.dtype == np.uint8:
            data_range = 255  # For 8-bit images (values between 0 and 255)
        elif original.dtype == np.float32 or original.dtype == np.float64:
            data_range = 1.0  # For floating point images (values between 0 and 1)
        else:
            raise ValueError(f"Unsupported image dtype: {original.dtype}")
        # SSIM returns a tuple: (score, full_ssim_map)
        ssim_score, _ = ssim(original, compressed, full=True, data_range=data_range)
        return ssim_score

    @staticmethod
    def overall_metrics(original, compressed):
        """Calculate and return both PSNR, SSIM, and storage savings metrics."""
        return {
            'PSNR': Metrics.calculate_psnr(original, compressed),
            'SSIM': Metrics.calculate_ssim(original, compressed),
        }

    def calculate_metrics_for_all(self):
        """Calculate metrics for all pairs of original and compressed images."""
        if len(self.original_images) != len(self.compressed_images):
            raise ValueError("The number of original and compressed images must be the same.")
        
        metrics_list = []

        for original, compressed in zip(self.original_images, self.compressed_images):
            # Convert images to numpy arrays if they are PIL images (for metric calculations)
            original_array = np.array(original)
            compressed_array = np.array(compressed)
            
            # Calculate metrics for each image pair
            metrics = self.overall_metrics(original_array, compressed_array)
            metrics_list.append(metrics)

        return metrics_list

    def metrics_flow(self):
        """Calculate metrics for all images and return the mean error (mean PSNR, mean SSIM) and storage savings."""
        # Calculate individual metrics and storage savings for all images
        metrics_list = self.calculate_metrics_for_all()
        
        # Extract PSNR, SSIM, and Storage Savings values for all images
        psnr_values = [metrics['PSNR'] for metrics in metrics_list]
        ssim_values = [metrics['SSIM'] for metrics in metrics_list]
        
        # Calculate the mean PSNR, mean SSIM, and mean Storage Savings
        mean_psnr = np.mean(psnr_values)
        mean_ssim = np.mean(ssim_values)
        
        # Round the values to 2 decimal places
        mean_psnr = round(mean_psnr, 3)
        mean_ssim = round(mean_ssim, 3)
        
        return {
            'Mean PSNR': mean_psnr,
            'Mean SSIM': mean_ssim,
        }
