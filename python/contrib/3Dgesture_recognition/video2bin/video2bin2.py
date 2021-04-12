"""
 Video to bin new 
"""

import cv2
import numpy as np
import os



FILEPATH = './test_actiontype7/'
OUTFILE = './Output/float/test_float32_actiontype7.bin'

CROP_SIZE = 112
CHANNEL_NUM = 3
CLIP_LENGTH = 16


def file_names():
"""
read file names
"""
    F = []
    for root, dirs, files in os.walk(FILEPATH):
        for file in files:  
            if os.path.splitext(file)[1] == '.jpg':
                F.append(root + file) 
    return F


def read_images(imglist):
"""
read images
"""
    imgArray = np.empty([1, CLIP_LENGTH, CROP_SIZE, CROP_SIZE, CHANNEL_NUM], dtype = np.float32)
    i = 0
    for img in imglist:
        image = cv2.imread(img)
        image = cv2.resize(image, (CROP_SIZE, CROP_SIZE))
        imgArray[0][i] = image 
        i = i + 1
        print(imgArray[0][i-1])

    return imgArray


imageNames = file_names()
imageNames.sort()
imgArray = read_images(imageNames)
imgArray.tofile(OUTFILE)
