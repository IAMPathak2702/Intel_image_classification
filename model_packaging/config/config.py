import pathlib
import os
import model_packaging
import datetime

# Get the root directory of the package
PACKAGE_ROOT = pathlib.Path(model_packaging.__file__).resolve().parent

# Define paths for data
DATAPATH = os.path.join(PACKAGE_ROOT, "data")
TRAIN_FILE = os.path.join(DATAPATH, "traindata")
EVAL_FILE = os.path.join(DATAPATH, "evaldata")
TEST_FILE = os.path.join(DATAPATH, "testdata")
# Using datetime to create a unique timestamp for the model directory


timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    

# Define the path to save the trained model
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'model', f"Model_{timestamp}")

# Define the name of the model file
MODEL_NAME = 'classification.pkl'

# Define class names for your classification task
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
