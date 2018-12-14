# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 20:24:17 2017

@author: Ana
"""
from PIL import Image,ImageOps, ImageChops, ImageEnhance
import os, sys
import numpy as np
import random
import colorsys
import preprocess as pr


rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

def shift_hue(arr, hout):
    r, g, b, a = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = h+hout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b, a))
    return arr

def colorize(image, hue):
    """
    Colorize PIL image `original` with the given
    `hue` (hue within 0-360); returns another PIL image.
    """
    img = image.convert('RGBA')
    arr = np.array(np.asarray(img).astype('float'))
    new_img = Image.fromarray(shift_hue(arr, hue/100.).astype('uint8'), 'RGBA')
    return new_img

def datacnn(source='cnn2',target='slije2'):
    j=0
    root=os.getcwd()+'\\'
    read=root+source+'\\'
    write=root+target+'\\'
    if not os.path.exists(write):
        os.makedirs(write)
    listing = os.listdir(read)
    for file in listing:
        im=Image.open(read+file)
        if (random.uniform(0.0,1.0)<0.2) & (j<500):
            j=j+1
            for i in range(20):
                width=im.size[0]
                height=im.size[1]
                temp=im  
                #ROTATION
                rot=random.uniform(-20, 20)
                temp=temp.rotate(rot)
                #CROP&SCALE
                a=random.randint(int(width/6*0.8),int(width/6*1.2))
                b=random.randint(int(height/6*0.8),int(height/6*1.2))
                w=random.uniform(0.8,1.2)
                z=random.uniform(0.8,1.2)
                temp=temp.crop((a,b,a+w*2*width/3,b+2*z*height/3))
                #HUE
                hue=random.randint(-5,5)   
                temp=colorize(temp,hue)
                #COLOR
                enhancer = ImageEnhance.Color(temp)
                col=random.uniform(0.5,2)
                temp=enhancer.enhance(col)
                #CONTRAST
                enhancer = ImageEnhance.Contrast(temp)
                contr=random.uniform(0.5,2)
                temp=enhancer.enhance(contr)
                
                temp=temp.resize((64,64))
                #SAVE
                temp.save(write+file[:-4]+'br'+str(i)+'.jpg')
datacnn('cnn3','slije2334')