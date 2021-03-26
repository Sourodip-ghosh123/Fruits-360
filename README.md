# Fruits-360

![workflow](https://user-images.githubusercontent.com/51257384/112630435-97825000-8e5b-11eb-82f8-46738c6cc351.png)

This is the official repository for "Supervised Learning based Neural Networks forFruit Identification" by Sourodip Ghosh, Md. Jashim Mondal, Sourish Sen, Nilanjan Kar Roy, and Suprava Patnaik

This repository contains the code used for analysis of CNN and ResNet50 V2 networks across images in Fruits-360 database. To know more / download images from Fruits-360 database, check this link: https://www.kaggle.com/moltean/fruits

Dataset properties
Total number of images: 90483.
Training set size: 67692 images (one fruit or vegetable per image).
Test set size: 22688 images (one fruit or vegetable per image).
Number of classes: 131 (fruits and vegetables).
Image size: 100x100 pixels.

We use 41 classes of fruits (28,283 images) from the Fruits-360 dataset. 

## Folder structure:

├── CNN
│   ├── cnn.ipynb
│   ├── cnn_model.py
│   ├── pre-processing.py
├── ResNet50 V2
│   ├── resnet50_v2.ipynb
│   ├── resnet50_v2_model.py
│   ├── pre-processing.py
└── README.md
