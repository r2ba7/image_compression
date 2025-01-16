import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import backend as K
import gc
import numpy as np

K.clear_session()
gc.collect()

class CNNCompression:
    def __init__(self, input_shape=None):
        """
        Initialize the CNN Compression Autoencoder.
        
        Args:
            input_shape (tuple): Shape of input images (height, width, channels).
                                 If None, will be determined from first training batch.
        """
        self.input_shape = input_shape
        self.latent_dim = 512
        self.model = None
        self.encoder = None
        self.decoder = None

    def advanced_loss(self, y_true, y_pred):
        """
        Combine MSE and SSIM into a simple loss function.
        """
        # Mean Squared Error (MSE)
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # Structural Similarity Index Loss (1 - SSIM)
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

        # Combine MSE and SSIM with simple weighting
        return mse_loss + 0.5 * ssim_loss

    def build_model(self):
        """
        Builds the CNN Autoencoder model for grayscale images.
        """
        # Input dimensions
        height, width = self.input_shape[:2]

        # ---- Encoder ----
        input_img = layers.Input(shape=self.input_shape)

        # Convolutional layers
        x = layers.Conv2D(32, (3, 3), activation=None, padding='same')(input_img)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)  # Halves dimensions

        x = layers.Conv2D(64, (3, 3), activation=None, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)  # Quarters dimensions

        x = layers.Conv2D(128, (3, 3), activation=None, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)  # Eighths dimensions

        # Latent space
        x = layers.Flatten()(x)
        latent = layers.Dense(self.latent_dim, activation='relu')(x)

        # Create encoder model
        self.encoder = models.Model(input_img, latent, name="Encoder")

        # ---- Decoder ----
        latent_input = layers.Input(shape=(self.latent_dim,))

        # Fully connected layer to reshape to decoder dimensions
        x = layers.Dense((height // 8) * (width // 8) * 128, activation='relu')(latent_input)
        x = layers.Reshape((height // 8, width // 8, 128))(x)

        # Transpose convolutional layers
        x = layers.Conv2DTranspose(128, (3, 3), activation=None, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D((2, 2))(x)

        x = layers.Conv2DTranspose(64, (3, 3), activation=None, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D((2, 2))(x)

        x = layers.Conv2DTranspose(32, (3, 3), activation=None, padding='same')(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D((2, 2))(x)

        # Final layer to reconstruct grayscale image
        decoded = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

        # Create decoder model
        self.decoder = models.Model(latent_input, decoded, name="Decoder")

        # ---- Full Autoencoder ----
        self.model = models.Model(input_img, self.decoder(self.encoder(input_img)), name="Autoencoder")

        # Compile the autoencoder
        self.model.compile(optimizer=Adam(), loss=self.advanced_loss)
        

    def train(self, train_images, batch_size=8, epochs=10):
        """
        Train the autoencoder on the training images.
        
        Args:
            train_images (numpy.ndarray): Training images 
            batch_size (int): Number of images per batch
            epochs (int): Number of training epochs
        """
        # Ensure correct input shape
        if len(train_images.shape) == 3:
            train_images = np.expand_dims(train_images, axis=-1)
        
        # Set input shape if not already set
        if self.input_shape is None:
            self.input_shape = train_images.shape[1:]
        
        # Build model with the specific input shape
        self.build_model()
        
        # Normalize images to [0, 1] range
        train_images = train_images.astype('float32') / 255.0
        
        self.model.fit(
            train_images, 
            train_images, 
            epochs=epochs, 
            batch_size=batch_size, 
            shuffle=True
        )

    def compress(self, images):
        """
        Compress a list or batch of images.
        
        Args:
            images (numpy.ndarray): Input images 
        
        Returns:
            numpy.ndarray: Compressed latent representations
        """
        # Ensure correct input shape
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
        
        # Normalize images to [0, 1] range
        images = images.astype('float32') / 255.0
        
        # Compress using encoder
        compressed = self.encoder.predict(images)
        return compressed

    def decompress(self, compressed_images):
        """
        Decompress latent representations back to images.
        
        Args:
            compressed_images (numpy.ndarray): Compressed latent representations
        
        Returns:
            numpy.ndarray: Reconstructed images
        """
        # Ensure compressed images are in the correct shape
        if len(compressed_images.shape) == 1:
            compressed_images = compressed_images.reshape(1, -1)
        
        # Decompress using decoder
        decompressed = self.decoder.predict(compressed_images)
        
        # Denormalize images back to original range
        decompressed = (decompressed * 255).astype(np.uint8)
        decompressed = decompressed.reshape(decompressed.shape[0], 200, 200)
        return decompressed

    def compress_then_decompress(self, images):
        """
        Compress and then decompress a batch of images.
        
        Args:
            images (numpy.ndarray): Input images
        
        Returns:
            numpy.ndarray: Reconstructed images
        """
        # Compress the images to latent representations
        compressed = self.compress(images)
        
        # Decompress the latent representations to reconstruct the images
        decompressed = self.decompress(compressed)
        return decompressed

    def save_model(self, path):
        """Save the trained model to a file."""
        self.model.save(path)

    def load_model(self, path):
        """Load a pre-trained model from a file."""
        self.model = models.load_model(path)
        # Recreate encoder and decoder
        self.encoder = models.Model(self.model.input, self.model.get_layer(index=6).output)
        self.decoder = models.Model(
            self.model.get_layer(index=7).input, 
            self.model.get_layer(index=-1).output
        )


class CNNCompressionPretrained:
    def __init__(self, input_shape=None, pretrained_model="efficientnet"):
        """
        Initialize the CNN Compression Autoencoder with a pre-trained encoder.
        
        Args:
            input_shape (tuple): Shape of input images (height, width, channels).
            pretrained_model (str): Pre-trained model to use ('efficientnet' or 'unet').
        """
        self.input_shape = input_shape
        self.latent_dim = 512
        self.pretrained_model = pretrained_model
        self.model = None
        self.encoder = None
        self.decoder = None

    def advanced_loss(self, y_true, y_pred):
        """
        Combine MSE and SSIM into a simple loss function.
        """
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        return mse_loss + 0.5 * ssim_loss

    def build_model(self):
        """
        Build the compression autoencoder model with a pre-trained encoder.
        """
        if self.pretrained_model == "efficientnet":
            print("efficientnet")
            encoder_base = EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_shape=self.input_shape
            )
            encoder_output = encoder_base.output
            encoder_output = layers.GlobalAveragePooling2D()(encoder_output)
            latent = layers.Dense(self.latent_dim, activation="relu")(encoder_output)

            # Define the encoder model
            self.encoder = models.Model(encoder_base.input, latent, name="Encoder")

        elif self.pretrained_model == "unet":
            print("unet")
            # Implement a U-Net encoder
            encoder_input = layers.Input(shape=self.input_shape)
            c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
            c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
            p1 = layers.MaxPooling2D((2, 2))(c1)
            
            c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
            c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
            p2 = layers.MaxPooling2D((2, 2))(c2)
            
            c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
            c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
            p3 = layers.MaxPooling2D((2, 2))(c3)

            # Bottleneck
            bottleneck = layers.Flatten()(p3)
            latent = layers.Dense(self.latent_dim, activation="relu")(bottleneck)

            # Define the encoder model
            self.encoder = models.Model(encoder_input, latent, name="Encoder")

        # ---- Decoder ----
        latent_input = layers.Input(shape=(self.latent_dim,))

        x = layers.Dense((self.input_shape[0] // 8) * (self.input_shape[1] // 8) * 128, activation="relu")(latent_input)
        x = layers.Reshape((self.input_shape[0] // 8, self.input_shape[1] // 8, 128))(x)

        x = layers.Conv2DTranspose(128, (3, 3), activation="relu", padding="same")(x)
        x = layers.UpSampling2D((2, 2))(x)

        x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")(x)
        x = layers.UpSampling2D((2, 2))(x)

        x = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same")(x)
        x = layers.UpSampling2D((2, 2))(x)

        decoded = layers.Conv2DTranspose(self.input_shape[2], (3, 3), activation="sigmoid", padding="same")(x)

        # Define the decoder model
        self.decoder = models.Model(latent_input, decoded, name="Decoder")

        # ---- Full Autoencoder ----
        self.model = models.Model(self.encoder.input, self.decoder(self.encoder.output), name="Autoencoder")
        self.model.compile(optimizer=Adam(), loss=self.advanced_loss)

    def train(self, train_images, batch_size=8, epochs=10):
        """
        Train the autoencoder on the training images.
        
        Args:
            train_images (numpy.ndarray): Training images
            batch_size (int): Number of images per batch
            epochs (int): Number of training epochs
        """
        if len(train_images.shape) == 3:
            train_images = np.expand_dims(train_images, axis=-1)

        if self.input_shape is None:
            self.input_shape = train_images.shape[1:]

        self.build_model()

        train_images = train_images.astype("float32") / 255.0

        self.model.fit(
            train_images,
            train_images,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
        )

    def compress(self, images):
        """
        Compress a list or batch of images.
        """
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)

        images = images.astype("float32") / 255.0
        compressed = self.encoder.predict(images)
        return compressed

    def decompress(self, compressed_images):
        """
        Decompress latent representations back to images.
        """
        if len(compressed_images.shape) == 1:
            compressed_images = compressed_images.reshape(1, -1)

        decompressed = self.decoder.predict(compressed_images)
        decompressed = (decompressed * 255).astype(np.uint8)
        return decompressed

    def compress_then_decompress(self, images):
        """
        Compress and then decompress a batch of images.
        """
        compressed = self.compress(images)
        decompressed = self.decompress(compressed)
        return decompressed
