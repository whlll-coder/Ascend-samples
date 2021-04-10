## Video to bin new 

import cv2
import numpy as np
import os



FILEPATH = './test_actiontype7/'
OUTFILE = './Output/float/test_float32_actiontype7.bin'
#OUTFILE = './Output/float32/test_actiontype1.bin'

CROP_SIZE = 112
CHANNEL_NUM = 3
CLIP_LENGTH = 16

def file_names():
    F = []
    for root, dirs, files in os.walk(FILEPATH):
        for file in files:
            #print file.decode('gbk')  
            if os.path.splitext(file)[1] == '.jpg':
                #print(root + file) 
                F.append(root + file) 
    return F


def read_images(imglist):
    imgArray = np.empty([1,CLIP_LENGTH, CROP_SIZE, CROP_SIZE ,CHANNEL_NUM], dtype = np.float32)
    i = 0
    for img in imglist:
        image = cv2.imread(img)
        image = cv2.resize(image, (CROP_SIZE, CROP_SIZE))
        imgArray[0][i] = image 
        i = i + 1
        print(imgArray[0][i-1])
        #print(image.shape)

    return imgArray


imageNames = file_names()
imageNames.sort()

print(imageNames)

imgArray = read_images(imageNames)
print(imgArray.shape)

print(imgArray)

imgArray.tofile(OUTFILE)