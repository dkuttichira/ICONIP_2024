# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 22:57:29 2023

@author: deept
"""

import numpy as np
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Load the feature matrix and label vector

A = np.load(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10train_2000relu.npy') 
A = A[:, np.any(A != 0, axis=0)] 
b = y_train
# Display the shape of A and b
print(f"Shape of feature matrix A: {A.shape}")
print(f"Shape of label vector b: {b.shape}")


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

def back_substitution(R, c):
    n = len(c)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = c[i]
        for j in range(i+1, n):
            x[i] -= R[i, j] * x[j]
        x[i] /= R[i, i]
    return x
unique_classes = np.unique(b)
print("Unique Classes")
print(unique_classes)
weights = []

for cls in unique_classes:
    print(cls)
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

print("Weights for each class:", weights)
w=np.array(weights)
np.save("qrweightcifar10_2000relu.npy",w)
