# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:19:18 2025

@author: deept
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
import random  # Ensure to import the random module
from keras.utils import to_categorical

# Set the random seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Set up paths to your dataset folders
train_dir = 'C:/Users/deept/.spyder-py3/swedish/Train/'  # Modify the path based on where your Train data is stored
val_dir = 'C:/Users/deept/.spyder-py3/swedish/Val/'    # Modify the path based on where your Val data is stored
test_dir = 'C:/Users/deept/.spyder-py3/swedish/Test/'    # Modify the path based on where your Test data is stored

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Use ImageDataGenerator for preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255   # Normalize images to [0, 1]
    #rotation_range=20,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True,
    #fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # 'categorical' for multi-class classification
    seed=seed_value  # Add the seed parameter to ensure reproducibility
)

# Load val data
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=seed_value  # Add the seed parameter to ensure reproducibility
)

# Load testing data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=seed_value  # Add the seed parameter to ensure reproducibility
)

# Load the feature matrix and label vector for Swedish Leaf dataset
A = np.load('C:/Users/deept/.spyder-py3/swedish/train_features_swedish_leaf-3.npy')
A = A[:, np.any(A != 0, axis=0)] 
# Ensure `b` matches the number of samples in A
# Only take the classes for the number of samples in A
b = np.array(train_generator.classes[:A.shape[0]])  # Make sure the number of labels matches A's samples

# Display the shape of A and b
print(f"Shape of feature matrix A: {A.shape}")
print(f"Shape of label vector b: {b.shape}")

# Define the Gram-Schmidt QR function
def gram_schmidt_qr(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            q = Q[:, i]
            R[i, j] = np.dot(q, v)
            v = v - R[i, j] * q
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R

# Define the back-substitution function
def back_substitution(R, c):
    n = len(c)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = c[i]
        for j in range(i+1, n):
            x[i] -= R[i, j] * x[j]
        x[i] /= R[i, i]
    return x

# Get the unique class labels
unique_classes = np.unique(b)
print("Unique Classes")
print(unique_classes)

# Initialize an empty list to store class weights
weights = []

# For each class, perform Gram-Schmidt QR decomposition
for cls in unique_classes:
    print(f"Processing class {cls}")

    # Create a new label vector where instances of the current class are marked as 1, and all others as 0
    b_current_class = np.where(b == cls, 1, 0)

    # Perform Gram-Schmidt QR factorization
    Q, R = gram_schmidt_qr(A)

    # Compute c = Q^T * b_current_class
    c = np.dot(Q.T, b_current_class)

    # Solve Rx = c to find x using back-substitution
    x = back_substitution(R, c)

    # Store the weights for the current class
    weights.append(x)

# Convert the weights list to a NumPy array and save
weights = np.array(weights)
np.save('C:/Users/deept/.spyder-py3/swedish/qr_weights_swedish_leaf-3.npy', weights)

print("Weights for each class have been saved.")