import tensorflow as tf
import keras
from keras.layers import Dense, Flatten
from keras.applications import ResNet50
from model_packaging.config import config
from model_packaging.processing.preprocessing import DataAugmentationLayer


class CustomResNetModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(CustomResNetModel, self).__init__()
        # Data Augmentation Layer
        self.data_augmentation = DataAugmentationLayer()
        # Load ResNet50 base model without the top layer
        self.resnet = ResNet50(input_shape=(224, 224, 3), include_top=False)
        # Freeze all layers in the base model
        for layer in self.resnet.layers:
            layer.trainable = False
        # Define the output layer for your specific classification task
        self.flatten = Flatten()
        self.output_layer = Dense(num_classes, activation="softmax", name="output_layer")

    def call(self, inputs, training=False):
        # Data augmentation
        if training:
            inputs = self.data_augmentation(inputs, training=training)
        # Forward pass
        x = self.resnet(inputs)
        x = self.flatten(x)
        output = self.output_layer(x)
        return output

# Instantiate the model
model = CustomResNetModel(num_classes=len(config.CLASS_NAMES))

# Summary of the model architecture
model.build(input_shape=(None, 224, 224, 3))  # Manually build the model to specify input shape


