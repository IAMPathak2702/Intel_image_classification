# Intel_image_classification

# Project Description

The project aims to develop a deep learning model for image classification using the TensorFlow and Keras libraries. The model is based on the ResNet50 architecture, a powerful convolutional neural network commonly used for image recognition tasks. The dataset used for training and evaluation consists of images belonging to six different classes: buildings, forest, glacier, mountain, sea, and street.

## Goals

1. **Model Training**: Implement and train a custom ResNet50 model using TensorFlow and Keras. The model will be trained on a labeled dataset of images to learn to classify them into the correct categories.

2. **Evaluation**: Evaluate the trained model's performance on a separate test dataset to assess its accuracy and generalization capabilities.

3. **Prediction**: Use the trained model to make predictions on new, unseen images. This will involve loading the model and inputting images to obtain predictions about their classes.

4. **Model Packaging**: Organize the project structure into directories for data storage, model packaging, and scripts for training, evaluation, and prediction. Ensure proper documentation and version control for reproducibility and collaboration.

Overall, the project aims to demonstrate the process of building, training, and deploying a deep learning model for image classification tasks using state-of-the-art techniques and best practices in machine learning engineering.

# Directory path
```
model_packaging
    ├───config
    ├───data
    │   ├───evaldata
    │   │   ├───buildings
    │   │   ├───forest
    │   │   ├───glacier
    │   │   ├───mountain
    │   │   ├───sea
    │   │   └───street
    │   ├───testdata
    │   └───traindata
    │       ├───buildings
    │       ├───forest
    │       ├───glacier
    │       ├───mountain
    │       ├───sea
    │       └───street
    ├───model
    └───processing
```

# Model Layer
<img src="https://raw.githubusercontent.com/IAMPathak2702/Intel_image_classification/main/images/model.png" alt="Model Image" style="width:100%;">


