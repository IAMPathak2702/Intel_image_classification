import os
import tensorflow as tf
from model_packaging.config import config

def load_dataset(filename:str, prefetch:bool=False, batch_size:int=32):
    """
    Loads dataset from the specified filename.

    Args:
        filename (str): Name of the dataset file.
        prefetch (bool): Whether to prefetch the dataset for better performance.
        batch_size (int): Batch size for the dataset.

    Returns:
        tf.data.Dataset: Loaded dataset.
    """
    try:
        filepath = os.path.join(config.DATAPATH, filename)
        if prefetch:
            _data = tf.keras.preprocessing.image_dataset_from_directory(
                filepath,
                label_mode="categorical",
                image_size=(224, 224),
                seed=42,
                batch_size=batch_size
            ).prefetch(tf.data.AUTOTUNE)
            return _data
        else:
            _data = tf.keras.preprocessing.image_dataset_from_directory(
                filepath,
                label_mode="categorical",
                image_size=(224, 224),
                seed=42,
                batch_size=batch_size
            )
        return _data
    except Exception as e:
        print(f"Error loading dataset from {filepath}: {e}")
        return None

def save_model(model_to_save, modelname):
    """
    Saves the model under the specified name.

    Args:
        model_to_save: Model object to be saved.
        modelname (str): Name of the model to be saved.

    Returns:
        None
    """
    try:
        save_path = os.path.join(config.SAVE_MODEL_PATH, modelname)
        model_to_save.save(save_path)
        print(f"Model has been saved under the name: {modelname}")
    except Exception as e:
        print(f"Error saving model to {save_path}: {e}")


def load_pipeline(model_name=config.MODEL_NAME):
    """
    Load a pre-trained model from the specified path.

    Args:
        model_name (str): Name of the model to be loaded.

    Returns:
        tf.keras.Model: Loaded model.
    """
    try: 
        load_path = os.path.join(config.SAVE_MODEL_PATH, model_name)
        # Load model (replace this line with your model loading logic)
        model_loaded = tf.keras.models.load_model(load_path)
        print(f"Model has been loaded")
        return model_loaded
    except Exception as e:
        print(f"Error loading pipeline from {model_name}: {e}")
        return None

# Usage example:
# loaded_model = load_pipeline("my_model")

# Usage example:
# 1. Loading dataset
# dataset = load_dataset("my_dataset", prefetch=True, batch_size=64)

# 2. Saving model
# save_model(my_model, "my_model")
