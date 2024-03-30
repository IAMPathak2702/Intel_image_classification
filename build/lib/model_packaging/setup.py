import io
import os
from pathlib import Path

from setuptools import find_packages, setup


# Metadata of package
NAME = 'Intel_Image_Classification'
DESCRIPTION = '''The project aims to develop a deep learning model for 
image classification using the TensorFlow and Keras libraries. 
The model is based on the ResNet50 architecture, a powerful 
convolutional neural network commonly used for image recognition tasks.
The dataset used for training and evaluation consists of images belonging 
to six different classes: buildings, forest, glacier, mountain, sea, and street.'''

URL = 'https://github.com/IAMPathak2702'
EMAIL = 'vp.ved.vpp@gmail.com'
AUTHOR = 'Ved Prakash Pathak'
REQUIRES_PYTHON = '>=3.7.0'

pwd = os.path.abspath(os.path.dirname(__file__))

# Get the list of packages to be installed
def list_reqs(fname='requirements.txt'):
    with io.open(os.path.join(pwd, fname), encoding='utf-8') as f:
        return f.read().splitlines()

try:
    with io.open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Load the package's __version__.py module as a dictionary.
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = os.path.join(ROOT_DIR, "prediction_model")
about = {}

with open(os.path.join(PACKAGE_DIR, 'VERSION')) as f:
    _version = f.read().strip()
    about['__version__'] = _version

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    package_data={'prediction_model': ['VERSION']},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)