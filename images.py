# -*- coding: utf-8 -*-
"""
Created on Mon May 16 09:38:33 2016

@author: frickjm
"""

import numpy as np
import scipy.misc as mi
from os import listdir

def getImage(imPath):
    return mi.imread(imPath)

def getRegion(image,size=128,downSamp=True,downSampFactor=2):
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
    thresholdVal= size**2*0.1
    
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
        print(loc)
        stop=raw_input(imFile)
    
    return region, loc
        

imDir     = "images/"
imFiles   = listdir(imDir)

for imFile in imFiles:
    imPath    = imDir+imFile
    image     = getImage(imPath)
    mi.imsave("wholeImage.jpg",image)
    region,loc= getRegion(image)
