# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 08:59:22 2022

@author: deept
"""
import numpy as np
from tensorflow.keras.models import load_model 
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import accuracy_score
import scipy.io

l=25 #Layer from which the feature is to be extracted
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train= (x_train.astype('float32'))/255
x_test= (x_test.astype('float32'))/255
model=load_model("cifar10model_2000relu.h5")


for i in range(len(model.layers)):
	layer = model.layers[i]

	print(i, layer.name, layer.output.shape)
    
new_model = Model(inputs=model.inputs, outputs=model.layers[l].output)

test_features=new_model.predict(x_test)
train_features=new_model.predict(x_train)

np.save(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10test_2000relu.npy',test_features)
np.save(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10train_2000relu.npy',train_features)

#scipy.io.savemat(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10_train.mat', {'x_train': x_train})
#scipy.io.savemat(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10_test.mat', {'x_test': x_test})
#scipy.io.savemat(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10_train_features_2n.mat', {'x_train_features_2n': train_features})
#scipy.io.savemat(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10_test_features_2n.mat', {'x_test_features_2n': test_features})
#scipy.io.savemat(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10_ytrain.mat', {'y_train': y_train})
#scipy.io.savemat(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10_ytest.mat', {'y_test': y_test})
l=np.sum(train_features,axis=0)
c=np.count_nonzero(l) 