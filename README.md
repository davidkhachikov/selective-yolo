# Selective Detection

## Collaborators
- Dmitry Nekrasov, B22-DS-02
- Anastasiia Shvets, B22-DS-02
- David Khachikov, B22-DS-02

## Project description
The goal of the project is to test Selective Detection - a model ensemble structure based on YOLO to confidently detect objects on images. The ensemble compares output of each checkpoint created during training with neighbouring checkpoints and with final model, and decides whether the image is attacked or not, and displays the image only if it is confident in predictions. To view how the model works visit [this notebook](https://github.com/davidkhachikov/selective-yolo/blob/main/notebooks/selective-detection-hypo-test.ipynb). Do not launch the code cells after "for Kaggle only"comments if you are not on Kaggle.

## Contents
### attacks
Folder with functions for attacks, attack visualizations and creating dataset with attacked images.

### checkpoint
Folder where checkpoints of trained model should be loaded.

### code
Folder with functions necessary for training and evaluation of checkpoints and ensemble.

### data
Folder with files containing summary of training process.

### notebooks
Folder with .ipynb notebooks contating experiments performed during the project

### resources
Folder containing miscellaneous files necessary for notebooks
