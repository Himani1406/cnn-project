# Potato Disease Classification using Convolutional Neural Networks

![Potato Disease Classification](path_to_image.png) <!-- Replace with an image relevant to your project -->

This repository contains the model and API for a potato disease classification project using Convolutional Neural Networks (CNNs). This project aims to accurately classify different diseases affecting potato plants based on images of their leaves. By utilizing deep learning techniques, specifically CNNs, we aim to create a robust and reliable classifier that can assist in early disease detection and improve crop management strategies.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

## Introduction
Potato crops are susceptible to diseases that can significantly impact yield and quality. Traditional methods of disease identification and management are often time-consuming and less accurate. This project focuses on automating the process using deep learning techniques, specifically CNNs, to classify different diseases based on leaf images.

## Dataset
We used a Kaggle dataset containing labeled images of healthy potato leaves as well as leaves affected by various diseases. The dataset can be found [here]([link_to_dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)).

## Installation
1. Clone this repository: `git clone https://github.com/your-username/your-repository.git`
2. Navigate to the project directory: `cd your-repository
3. Install the required dependencies: `pip install -r requirements.txt`


## Model Architecture
Our CNN model consists of several convolutional layers followed by max-pooling layers. The architecture is designed to extract hierarchical features from the input images, enabling the network to learn discriminative patterns related to different diseases.




## Training
During training, the dataset is split into training and validation sets. The model is trained using the training set and validated on the validation set. We use the Adam optimizer and categorical cross-entropy loss for training.

## Evaluation
The model's performance is evaluated using accuracy, precision, recall, and F1-score metrics on the test set. Confusion matrices and classification reports provide insights into the model's ability to classify different diseases.

## Results
Our trained model achieved pretty high accuracy on the validation set.

![image](https://github.com/Himani1406/cnn-project/assets/114576874/a7358985-e05f-4b3c-a79d-ca3fd280bc11)



## Future Enhancements
- Experiment with different CNN architectures (e.g., VGG, ResNet) for improved performance.
- Implement data augmentation techniques to enhance the model's generalization further.
- Explore transfer learning by fine-tuning pre-trained models on this dataset.
- Develop a user-friendly web interface for real-world disease classification.



