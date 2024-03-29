import tensorflow as tf
import keras
from keras.layers import Dense, Flatten
from keras.applications import ResNet50
from model_packaging.config import config
from model_packaging.processing.preprocessing import DataAugmentationLayer
from keras import Model


def create_resnet_model(input_shape, num_classes, class_names):
    """
    Create a ResNet model with a custom output layer.

    Args:
        input_shape (tuple): Input shape of the images (e.g., (224, 224, 3)).
        num_classes (int): Number of output classes.
        class_names (list): List of class names.

    Returns:
        tf.keras.Model: The ResNet model with a custom output layer.
    """
    # Load pre-trained ResNet50 model without top layer
    resnet = ResNet50(input_shape=input_shape, include_top=False)

    # Freeze all layers in the base ResNet model
    for layer in resnet.layers:
        layer.trainable = False

    # Flatten the output of ResNet
    x = Flatten()(resnet.output)
    
    # Create custom output layer
    output_layer = Dense(num_classes, activation="softmax", name="output_layer")(x)

    # Combine ResNet base model and custom output layer
    model = Model(resnet.input, output_layer, name="RESNET_MODEL")

    return model




