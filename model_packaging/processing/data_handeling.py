import os
import tensorflow as tf
from model_packaging.config import config
from PIL import Image
import random
import matplotlib.image as mpimg

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



def prepare_image(image_path):
    """
    Load and preprocess an image for prediction.

    Args:
        image_path (str): Path to the folder containing image files.

    Returns:
        tf.Tensor: Preprocessed image tensor.
    """
    # List all files in the folder
    files = os.listdir(image_path)
    
    # Filter out non-image files (e.g., directories)
    image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    # Pick a random image file
    random_image_file = random.choice(image_files)
    
    # Construct the full path to the randomly selected image file
    random_image_path = os.path.join(image_path, random_image_file)
    
    # Load the image using PIL
    image = mpimg.imread(random_image_path)
    
    image = tf.image.resize(image, size=(224, 224))

    # Resize the image to 224x224
    
    
    # Convert the image to a TensorFlow tensor
    image_tensor = tf.convert_to_tensor(image)
    # Expand dimensions to add a batch dimension
    image_tensor = tf.expand_dims(image_tensor, axis=0)
 
    
    return image_tensor