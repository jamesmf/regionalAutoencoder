# -*- coding: utf-8 -*-
"""
Created on Mon May 16 09:38:33 2016

@author: frickjm
"""

import numpy as np
import scipy.misc as mi
from os import listdir


def getModel(img_width, img_height, out_size):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))   
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

def getImage(imPath):
    return mi.imread(imPath)*1./255

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
        print("check blank: ",np.sum(region), thresholdVal)
        if np.sum(region) > thresholdVal:
            isBlank = False
            #mi.imsave("big.jpg",region)
            if downSamp:
                region = mi.imresize(region,(newSize,newSize),interp='bilinear')
            mi.imsave("small.jpg",region)
    print(loc)
    stop=raw_input(imFile)
    
    return region, loc
        

imDir     = "images/"
imFiles   = listdir(imDir)

for imFile in imFiles:
    imPath    = imDir+imFile
    image     = getImage(imPath)
    mi.imsave("wholeImage.jpg",image)
    region,loc= getRegion(image,downSamp=False)
