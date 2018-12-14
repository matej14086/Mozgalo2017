# -*- coding: utf-8 -*-
"""
Created on Sat May 20 22:53:24 2017

@author: Petar
"""
from num_cluster import cal_score
from PIL import Image,ImageOps, ImageChops, ImageEnhance
import os, sys
import preprocess as pr
from keras.models import load_model
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.cluster import KMeans

def toarray(source='final'):
    y=np.zeros((10000,500),dtype=np.int)
    x=np.zeros((10000,64,64,3),dtype=np.int)
    root=os.getcwd()+'\\'
    read=root+source+'\\'
    i=0
    listing = os.listdir(read)
    for file in listing:
        im = Image.open(read + file)
        x[i,]=np.array(im)
        y[i,int(i/20)]=1
        i=i+1
    return (x,y)
def toarray2(source='cnn'):
    x=[]
    root=os.getcwd()+'\\'
    read=root+source+'\\'
    i=0
    listing = os.listdir(read)
    for file in listing:
        im = Image.open(read + file)
        x.append(np.array(im))
    return np.array(x)

def train(source='cnn',name_model='bzvz.h5'):
    (X_train, Y_train) =toarray(source)
    
    
    X_train = X_train.astype('float32')
    
    X_train /= 255
    model.summary()
    
    model = Sequential()
     
    model.add(Conv2D(64, (5,5), activation='relu', input_shape=(64,64,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1000, activation='softmax'))
    
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    
    model.fit(X_train, Y_train, batch_size=64, nb_epoch=20, verbose=1)
    model.save(name_model)




def knn_output(source='cnn',name_model='my_model2.h5'):
    model = load_model(name_model)
    X_test =toarray2('cnn')
    X_test = X_test.astype('float32')
    
    X_test /= 255
    intermediate3_layer_model = Model(inputs=model.input,outputs=model.layers[5].output)
    intermediate3_output = intermediate3_layer_model.predict(X_test)
    np.save('feature_c64-c64-128',intermediate3_output)
def clustering(path="feature_c64-c64-128.npy"):
	data=np.load(path)
	a,b=np.shape(data)
	for i in range (b):
		std=np.std(data[:,i])
		sr=np.mean(data[:,i])
		data[:,i]=(data[:,i]-sr)/std
		if(std==0 and sr==0):
			data[:,i]=0
	Kmin=2
	Kmax=20
	N_opt=np.int(cal_score(data,Kmin,Kmax))

	kmeans_model=KMeans(n_clusters=N_opt).fit(data)
	labels=kmeans_model.labels_
	np.save('labels_c64-c64-128',labels)
