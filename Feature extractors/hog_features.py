# -*- coding: utf-8 -*-
"""
Created on Thu May 11 00:51:29 2017

@author: Matej
"""
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from sklearn import svm
from skimage.feature import hog
from CNN_AE import Import_data
orientations= 9
pixels_per_cell= [8, 8]
cells_per_block=[2, 2]
visualize= False
normalize= True

def hogftr(source="for_encoder"):
	data=Import_data(source)
	hist=np.zeros((len(data),1764))
	for i in range(len(data)):
		data[i,:,:,0]=data[i,:,:,0]*0.299+0.587*data[i,:,:,1]+0.114*data[i,:,:,2]
		hist[i,:]=hog(np.array(data[i,:,:,0]), orientations, pixels_per_cell, cells_per_block, visualize, normalize)
			
	write=os.getcwd()+'\\'+"model"+'\\'
	if not os.path.exists(write):
	          os.makedirs(write) 
	
	np.save(write+"hist",hist)

def visual_hog(path):
	visualize=True
	im=Image.open(path)

	im=np.array(im)

	fd,image=hog(np.array(im[:,:,0]), orientations, pixels_per_cell, cells_per_block, visualize, normalize)
	plt.imshow(image,cmap=plt.cm.gray)

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

