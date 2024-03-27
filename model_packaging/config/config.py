import pathlib
import os
import model_packaging

PACKAGE_ROOT = pathlib.Path(model_packaging.__file__).resolve().parent
DATAPATH = os.path.join(PACKAGE_ROOT, "data")
TRAIN_FILE = os.path.join(DATAPATH, "train")
EVAL_FILE = os.path.join(DATAPATH, "eval")
TEST_FILE = os.path.join(DATAPATH, "test")
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'model')


MODEL_NAME = 'classification.pkl'


CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']