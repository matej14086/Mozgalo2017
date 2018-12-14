# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:00:00 2017

@author: Matej
"""
import os
import skfuzzy as fuzz
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
from sklearn.cluster import KMeans
import math

def validity_score(data,labels,centroid):
	intra=0
	for i in range(max(labels)):
		for j in range(len(labels)):
			if(labels[j]==i):
				intra=intra+np.dot(data[j,:],centroid[i,:])
	intra=intra/len(labels)
	inter=math.inf
	for i in range(max(labels)):
		for j in range(i+1,max(labels)):
			temp=np.dot(centroid[i,:],centroid[j,:])
			if(temp<inter):
				inter=temp
	return intra/inter

def cal_score(data,minK,maxK):
	a,b=np.shape(data)
	for i in range (b):
		std=np.std(data[:,i])
		sr=np.mean(data[:,i])
		data[:,i]=(data[:,i]-sr)/std
		if(std==0 and sr==0):
			data[:,i]=0

	score=[]
	x=[]

	for i in range (minK,maxK):
	    kmeans_model = KMeans(n_clusters=i, random_state=4).fit(data)
	    labels = kmeans_model.labels_
	    score.append(validity_score(data,labels,kmeans_model.cluster_centers_))
	    x.append(i)

	plt.plot(x,score)
	for i in range(1,len(score)-1):
		if(score[i-1]<score[i] and score[i]>score[i+1]):
			mini=score[i]
			N_opt=i
			for j in range(i,len(score)):
				if(score[j]<mini):
					mini=score[j]
					N_opt=j
			return N_opt+minK
	return 0
