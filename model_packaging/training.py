import tensorflow as tf

from model_packaging.config import config
from model_packaging.processing.data_handeling import load_dataset ,save_model
from model_packaging.pipeline import CustomResNetModel ,Model







def perform_training():
    model = CustomResNetModel()
    train_dataset = load_dataset(config.TRAIN_FILE, prefetch=True, batch_size=32)  # Assuming you have a function to load your dataset

    # Train the model with data augmentation
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=10, verbose=1)
    
if __name__=='__main__':
    perform_training()