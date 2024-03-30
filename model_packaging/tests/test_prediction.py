import pytest
import tensorflow as tf
from model_packaging.config import config
from model_packaging.processing.data_handeling import prepare_image, load_pipeline
from model_packaging.predict import generate_predictions
import warnings

def pytest_configure(config):
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow.*")

@pytest.fixture
def single_prediction(filepath: str=config.SAVE_MODEL_PATH, class_names: list=config.CLASS_NAMES) -> str:
    """
    Perform prediction on a single image using the provided model.

    Args:
        filepath (str): Path to the image file.
        class_names (list): List of class names.

    Returns:
        str: Predicted class name.
    """
    # Load the model
    model = load_pipeline(filepath)  

    # Load and preprocess the image
    img = prepare_image(config.TEST_FILE)

    # Make predictions on the preprocessed image
    predictions = model.predict(img)

    # Get the index of the class with the highest probability prediction
    predicted_class_index = tf.argmax(predictions[0]).numpy()

    # Map the index to the corresponding class name
    predicted_class_name = class_names[predicted_class_index]

    # Return the predicted class name
    return predicted_class_name 

def test_single_pred_not_none(single_prediction) -> None:
    """
    Test function to check if the prediction result is not None.

    Args:
        single_prediction: The single_prediction function to be tested.

    Returns:
        None
    """
    assert single_prediction is not None
    
def test_single_pred_str_type(single_prediction) -> None:
    """
    Test function to check if the prediction result is of string type.

    Args:
        single_prediction: The single_prediction function to be tested.

    Returns:
        None
    """
    assert isinstance(single_prediction, str)

import pytest
import tensorflow as tf

