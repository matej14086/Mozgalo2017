# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:20:54 2017

@author: Matej
"""
import numpy as np
from pathlib import Path
from PIL import Image,ImageOps, ImageChops
import os
import matplotlib.pyplot as plt
import scipy.misc
from num_cluster import cal_score, bad_cluster
from sklearn.cluster import KMeans
from keras.layers import Input, Dense,Conv2DTranspose, Conv2D, MaxPooling2D, UpSampling2D, core, Flatten, LocallyConnected1D, Lambda, merge, Conv1D, Dropout
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model

ime="32-64-128-150"


def ImportCol(source):   
	
	root=os.getcwd()+'\\'
	read=root+source+'\\'
	listing = os.listdir(read)


	dataset=np.zeros((len(listing),64,64,3))
	i=0    
	for file in listing:
		my_file = Path(read+file)
		if my_file.is_file():
			
			im=Image.open(read+file)
			img_matrix=np.array(im)
			if (np.shape(img_matrix)[2]==3):
				dataset[i,:,:,:]=img_matrix
				i=i+1
		
	
	return dataset  
def Import_data(dest='for_encoder'):
    x_train=ImportCol(dest)
    x_train = x_train.astype('float32') / 255.
    return x_train

def arh(img):
   
    x = Conv2D(16, (3, 3),strides=2, activation='relu', padding='same',input_shape=(None,64,64,3),name="first")(img)
    x = MaxPooling2D((2, 2),strides=2, padding='same')(x)
    x = Conv2D(32, (3, 3),strides=1, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2),strides=2, padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2),strides=2, padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2),strides=2, padding='same')(x) 
    x=Flatten()(x)
    #x=Dropout(rate=0.3)(x)
    x=Dense(512,activation='relu')(x)        
    x=Dense(150,activation='relu',name="ext")(x)
    x=Dense(512,activation='relu')(x)              
    x=core.Reshape((2,2,128))(x)   
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(64, (3, 3),strides=1, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same',strides=1)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(16,(3, 3) , strides=1, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2DTranspose(3, (3, 3),strides=2, activation='sigmoid', padding='same')(x)
    return decoded

    

    

def train(dest='for_encoder'):
    write=os.getcwd()+'\\'+"model"+'\\'
    input_img = Input(shape=(64, 64, 3))  # adapt this if using `channels_first` image data format
    output_img =arh(input_img)
    autoencoder = Model(input_img, output_img)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    x_train=Import_data(dest)
    if os.path.exists(write+ime+'.h5'):
       autoencoder=load_model(write+ime+'.h5')
    filepath=write+ime+'.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,save_weights_only=False,period=1)
    callbacks_list = [checkpoint,TensorBoard(log_dir='/tmp/autoencoder')]
    autoencoder.summary()
    autoencoder.fit(x_train, x_train,epochs=100,
                    batch_size=10, shuffle=True,
                    validation_data=(x_train[:100], x_train[:100]),
                    callbacks= callbacks_list)    
        
    
def visual(dest='for_encoder'):
    
    
    load=os.getcwd()+'\\'+"model"+'\\'
    save=os.getcwd()+'\\'+"visual"+'\\'
    if not os.path.exists(save):
            os.makedirs(save)    
    input_img = Input(shape=(64, 64, 3))  # adapt this if using `channels_first` image data format
    output_img =arh(input_img)
    autoencoder = Model(input_img, output_img)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    x_train=Import_data(dest)
    autoencoder=load_model(load+ime+'.h5')
    decoded_imgs = autoencoder.predict(x_train)
    read=os.getcwd()+'\\'+dest+'\\'
    listing = os.listdir(read)
    n = len(x_train)
    
    for i in range(n):
        temp=np.zeros((64,64,3))
        temp[:,:,0]=decoded_imgs[i,:,:,0].reshape(64, 64)*255
        temp[:,:,1]=decoded_imgs[i,:,:,1].reshape(64, 64)*255
        temp[:,:,2]=decoded_imgs[i,:,:,2].reshape(64, 64)*255
        scipy.misc.imsave(save+listing[i],temp)
def extract(dest='for_encoder'):
    x_train=Import_data(dest)
    
    load=os.getcwd()+'\\'+"model"+'\\'
    autoencoder = load_model(load+ime+'.h5')
    autoencoder.summary()    
    midll = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("ext").output)
    decoded_imgs = midll.predict(x_train)
    np.save(load+"feature_of_"+ime,np.array(decoded_imgs))
	    
def filter_visual(path):
	load=os.getcwd()+'\\'+"model"+'\\'
	im=Image.open(path)
	im=im.resize((64,64))
	im=np.array(im)
	im=np.reshape(im,(1,64,64,3))
	autoencoder=load_model(load+ime+'.h5')
	filters=Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("first").output)
	output=filters.predict(im)
	fig=plt.figure()
	
	for i in range(4):
		for j in range(8):
			temp=fig.add_subplot(4,8,8*i+j+1)
			temp.imshow(output[0,:,:,8*i+j])
			temp.axis("off")

	plt.show()

def clustering(path="model\\feature_of_32-64-128-150.npy"):
	data=np.load(path)
	a,b=np.shape(data)
	for i in range (b):
		std=np.std(data[:,i])
		sr=np.mean(data[:,i])
		data[:,i]=(data[:,i]-sr)/std
		if(std==0 and sr==0):
			data[:,i]=0
	Kmin=2
	Kmax=30
	N_opt=np.int(cal_score(data,Kmin,Kmax))

	kmeans_model=KMeans(n_clusters=N_opt).fit(data)
	labels=kmeans_model.labels_
	if b<1700:
		labels=bad_cluster(labels,data)
	return labels



