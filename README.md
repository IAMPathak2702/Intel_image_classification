# Intel_image_classification

# Project Description

The project aims to develop a deep learning model for image classification using the TensorFlow and Keras libraries. The model is based on the ResNet50 architecture, a powerful convolutional neural network commonly used for image recognition tasks. The dataset used for training and evaluation consists of images belonging to six different classes: buildings, forest, glacier, mountain, sea, and street.

## Goals

1. **Model Training**: Implement and train a custom ResNet50 model using TensorFlow and Keras. The model will be trained on a labeled dataset of images to learn to classify them into the correct categories.

2. **Evaluation**: Evaluate the trained model's performance on a separate test dataset to assess its accuracy and generalization capabilities.

3. **Prediction**: Use the trained model to make predictions on new, unseen images. This will involve loading the model and inputting images to obtain predictions about their classes.

4. **Model Packaging**: Organize the project structure into directories for data storage, model packaging, and scripts for training, evaluation, and prediction. Ensure proper documentation and version control for reproducibility and collaboration.

Overall, the project aims to demonstrate the process of building, training, and deploying a deep learning model for image classification tasks using state-of-the-art techniques and best practices in machine learning engineering.


## DESCRIPTION 

- The "Intel Image Classification" project comprehensively demonstrates MLOps practices and tools. It encompasses the development of a deep learning model for image classification using TensorFlow and Keras, specifically based on the ResNet50 architecture.

- In addition to model development, this project focuses on the complete lifecycle management of the model through MLOps practices. The model is packaged using setuptools, facilitating easy distribution and deployment. Continuous integration and deployment (CI/CD) pipelines are established using Jenkins, ensuring automated testing and deployment of new model versions.

- For model monitoring and observability, Grafana is integrated to visualize key performance metrics and monitor model behavior in real-time. This ensures proactive identification of issues and performance degradation.

- Furthermore, the project showcases user interface (UI) implementations for interacting with the model. Streamlit is utilized to create interactive web applications for model demonstration and exploration, allowing users to upload images and receive predictions from the model in real-time. FastAPI is employed to develop a robust and scalable API endpoint for serving model predictions, enabling seamless integration with other systems and services.

- By incorporating these components, the "Intel Image Classification" project demonstrates a holistic approach to MLOps, encompassing model development, packaging, deployment, monitoring, and user interface development, showcasing proficiency in modern MLOps practices and tools.




# Directory path
```
C:.
├───config
│   └───__pycache__
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
│   ├───Model_2024-03-29_20-38-1
│   │   ├───assets
│   │   └───variables
│   └───ResNetModel
│       ├───assets
│       └───variables
└───processing
    └───__pycache__
```

# Model Layer
<img src="https://raw.githubusercontent.com/IAMPathak2702/Intel_image_classification/main/images/model.png" alt="Model Image" style="width:100%;">


