# -*- coding: utf-8 -*-
"""
Created on Mon May 16 09:38:33 2016

@author: frickjm
"""

import numpy as np
import scipy.misc as mi
from os import listdir

from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Merge

from keras.preprocessing.image import img_to_array, ImageDataGenerator


def getModel(img_width, img_height, out_size):
    bigModel = Sequential()
    bigModel.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))   
    bigModel.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    bigModel.add(ZeroPadding2D((1, 1)))
    bigModel.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    bigModel.add(MaxPooling2D((2, 2), strides=(2, 2)))
    bigModel.add(Dropout(0.5))
        
    bigModel.add(ZeroPadding2D((1, 1)))
    bigModel.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    bigModel.add(ZeroPadding2D((1, 1)))
    bigModel.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    bigModel.add(MaxPooling2D((2, 2), strides=(2, 2)))
    bigModel.add(Dropout(0.5))
    
    bigModel.add(ZeroPadding2D((1, 1)))
    bigModel.add(Convolution2D(128, 3, 3, activation='relu', name='conv3_1'))
    bigModel.add(ZeroPadding2D((1, 1)))
    bigModel.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    bigModel.add(MaxPooling2D((2, 2), strides=(2, 2)))
    bigModel.add(ZeroPadding2D((1, 1)))
    bigModel.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    bigModel.add(MaxPooling2D((2, 2), strides=(2, 2)))

    bigModel.add(Flatten())
    

    windowModel   = Sequential()
    windowModel.add(Dense(128,input_shape=(2,)))
    windowModel.add(Activation('relu'))
    
    model   = Sequential()
    model.add(Merge([bigModel, windowModel],mode='concat', concat_axis=-1))
    model.add(Dense(3*out_size**2))

    model.compile(optimizer='adadelta',loss='mse')
    
    return model


def getImage(imPath):
    return mi.imread(imPath)

def getRegion(image,size=64,downSamp=True,downSampFactor=2):
    """returns a random, non-blank region from an image.
    
        input:
            image - the source image
            size  - the number of pixels the subregion will be (on each side)
            
        output:
            region - the square region extracted
            loc    - the location of the image (relative to the size of input)
                     in the form [y, x] - e.g. [0.4, 0.6]
    """

    isBlank     = True    #don't want to select a blank region
    thresholdVal= size**2*0.05
    
    if downSamp:
        newSize     = int(size)/downSampFactor
    
    while isBlank:
        #upper left point of the candidate region
        loc = [np.random.rand(),np.random.rand()]
        upperLeft = [int((image.shape[0]-size)*loc[0]), int((image.shape[1]-size)*loc[1])]
        region = image[upperLeft[0]:upperLeft[0]+size,upperLeft[1]:upperLeft[1]+size]
        if np.sum(region) > thresholdVal:
            isBlank = False
            #mi.imsave("big.jpg",region)
            if downSamp:
                region = mi.imresize(region,(newSize,newSize),interp='bilinear')
            mi.imsave("small.jpg",region)
    #print(loc)
    
    return region.flatten(), loc
        

imDir       = "images/"
imFiles     = listdir(imDir)
numImages   = len(imFiles)
imSize      = 128
outsize     = 32
batchSize   = 16
numEpochs   = 10

model       = getModel(imSize,imSize,outsize)

for epoch in range(0,numEpochs):
    Xtrain  = np.zeros((numImages,3,imSize,imSize))
    XtrainL = np.zeros((numImages,2))
    ytrain  = np.zeros((numImages,3*outsize**2))
    for count,imFile in enumerate(imFiles):
        imPath  = imDir+imFile
        image   = getImage(imPath)
        image   = mi.imresize(image,(imSize,imSize))
        #mi.imsave("wholeImage.jpg",image)
        region,loc= getRegion(image,downSamp=True)
        image   = img_to_array(image)
        Xtrain[count,:,:,:] = image
        XtrainL[count,:] = loc
        ytrain[count,:] = region

    model.fit([Xtrain,XtrainL],ytrain,nb_epoch=20)
