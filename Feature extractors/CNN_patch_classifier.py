# -*- coding: utf-8 -*-
"""
Created on Sun May 21 00:19:54 2017

@author: Petar
"""

import keras
from keras.preprocessing import image
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
import numpy as np


from PIL import Image
import os, sys


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




def train(source='set',name_model='neki.h5'):
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False)
    
    train_generator = train_datagen.flow_from_directory(
            './'+source,
            target_size=(64, 64),
            batch_size=10,
            class_mode='categorical')
    
    
    model = Sequential()
     
    model.add(Conv2D(64, (3,3), activation='relu', input_shape=(64,64,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])



    model.fit_generator(train_generator,300, epochs=10)

    model.save(name_model)


def predict(source='cnn',name_model='finalcnn6.h5'):
    
    model=load_model(name_model)
    X_test =toarray2(source)
    X_test = X_test.astype('float32')
    
    X_test /= 255
    Y=model.predict(X_test)
    label=Y.argmax(axis=1)
    np.save('finalcnn_label',label)

