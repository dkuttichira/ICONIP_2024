# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 23:23:52 2023

@author: deept
"""
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import precision_score

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
y_train_check=y_train
y_test_check=y_test
x_train=np.load(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10train_2000relu.npy')
l=np.sum(x_train,axis=0) 
index=np.where(l==0)[0]
x_train=np.delete(x_train,index,axis=1)
x_test=np.load(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10test_2000relu.npy')
x_test=np.delete(x_test,index,axis=1)
weights=np.load("qrweightcifar10_2000relu.npy")

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

optimizer='adam'
input_dim = 1581
output_dim = 10 

def classification_model():
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

#training
keras_model = classification_model()

l=[]
x=weights.transpose() #weights
y=np.array([0]*10) #array of biases
l.append(x)
l.append(y)
keras_model.layers[0].set_weights(l)

prediction_class_train=keras_model.predict_classes(x_train)
prediction_class_test=keras_model.predict_classes(x_test)

accuracy_train=accuracy_score(prediction_class_train, y_train_check)
accuracy_test=accuracy_score(prediction_class_test, y_test_check)

# precision_train=precision_score(prediction_class_train, y_train_check,average='macro')
# precision_test=precision_score(prediction_class_test, y_test_check,average='macro')