import logging
import tensorflow as tf
from model_packaging.config import config
from model_packaging.processing.data_handeling import load_dataset ,save_model
from model_packaging.pipeline import create_resnet_model
from model_packaging.processing.preprocessing import ModelCallbacks
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO

def perform_training():
    # Assuming you've defined the model, train_dataset, and callbacks earlier

    # Load the model
    model = create_resnet_model(input_shape=(224, 224, 3), num_classes=len(config.CLASS_NAMES), class_names=config.CLASS_NAMES)

    # Load the training dataset
    train_dataset = load_dataset(config.TRAIN_FILE, prefetch=True, batch_size=32)  
    test_dataset = load_dataset(config.EVAL_FILE, prefetch=True, batch_size=32)  

    # Train the model with data augmentation
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    callbacks = ModelCallbacks()
    checkpoint_callback = callbacks.create_model_checkpoint(filepath=config.SAVE_MODEL_PATH)
    early_stopping_callback = callbacks.early_stopping()

    # Train the model
    model.fit(train_dataset,
              epochs=10,
              verbose=1,
              callbacks=[checkpoint_callback, early_stopping_callback],
              validation_data=test_dataset,
              use_multiprocessing=True)

if __name__=='__main__':
    perform_training()