import tensorflow as tf
import datetime
from model_packaging.config import config


class DataAugmentationLayer(tf.keras.layers.Layer):

    """
    Custom layer for data augmentation using TensorFlow.
    
    This layer applies various data augmentation techniques to the input images during training.

    Args:
        None
    
    Returns:
        Augmented images if training=True, otherwise returns the original images.
    """
    def __init__(self, **kwargs):
        """
        Initializes the DataAugmentationLayer.
        """
        super(DataAugmentationLayer, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        """
        Applies data augmentation to the input images during training.

        Args:
            inputs: Tensor containing the input images.
            training: Boolean flag indicating whether the model is in training mode.

        Returns:
            Augmented images if training=True, otherwise returns the original images.
        """
        if training:
            # Apply data augmentation here
            augmented_images = self.apply_data_augmentation(inputs)
            return augmented_images
        return inputs

    def apply_data_augmentation(self, images):
        """
        Applies a variety of augmentation techniques to the input images.

        Args:
            images: Tensor containing the input images.

        Returns:
            Augmented images after applying various augmentation techniques.
        """
        # Randomly apply a variety of augmentation techniques
        augmented_images = tf.image.random_flip_left_right(images)
        augmented_images = tf.image.random_flip_up_down(augmented_images)
        augmented_images = tf.image.random_contrast(augmented_images, 0.8, 1.2)
        augmented_images = tf.image.random_brightness(augmented_images, 0.1)
        augmented_images = tf.image.random_saturation(augmented_images, 0.8, 1.2)
        augmented_images = self.random_zoom(augmented_images)
        augmented_images = self.random_crop(augmented_images)
        return augmented_images

    def random_zoom(self, images):
        """
        Randomly zooms the input images.

        Args:
            images: Tensor containing the input images.

        Returns:
            Zoomed images after applying random zoom.
        """
        # Randomly zoom images
        zoom_factor = tf.random.uniform([], 0.8, 1.0)
        zoomed_images = tf.image.central_crop(images, zoom_factor)
        return zoomed_images

    def random_crop(self, images):
        """
        Randomly crops the input images.

        Args:
            images: Tensor containing the input images.

        Returns:
            Cropped images after applying random crop.
        """
        # Randomly crop images
        cropped_images = tf.image.random_crop(images, size=[tf.shape(images)[0], 24, 24, 3])
        return cropped_images




class ModelCallbacks:
    @staticmethod
    def create_model_checkpoint(filepath):
        """
        Create a ModelCheckpoint callback.

        Args:
            PACKAGE_ROOT (str): Root directory of the package.

        Returns:
            tf.keras.callbacks.ModelCheckpoint: ModelCheckpoint callback object.
        """
        # Define the directory where you want to save the model
        

        # Create a timestamp string using the current datetime
        

        # Create the ModelCheckpoint callback
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=0
        )
        return model_checkpoint_callback
    
    @staticmethod
    def early_stopping():
        """
        Create an EarlyStopping callback.

        Returns:
            tf.keras.callbacks.EarlyStopping: EarlyStopping callback object.
        """
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        return early_stopping_callback

# Example usage:

