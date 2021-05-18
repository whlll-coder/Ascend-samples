#!/usr/bin/env python
# encoding: utf-8
import sys
import os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path, ".."))
sys.path.append(os.path.join(path, "../../../common/"))
sys.path.append(os.path.join(path, "../../../common/atlas_utils"))

import numpy as np
import acl
import base64
import atlas_utils.utils as utils
from PIL import Image, ImageDraw, ImageFont
from atlas_utils.acl_dvpp import Dvpp
import atlas_utils.constants as const
from atlas_utils.acl_model import Model
from atlas_utils.acl_image import AclImage
from atlas_utils.acl_resource import AclResource

import cv2 
SRC_PATH = os.path.realpath(__file__).rsplit("/", 1)[0]
MODEL_PATH = os.path.join(SRC_PATH, "../model/vessel.om")
MODEL_WIDTH = 512
MODEL_HEIGHT = 512
INPUT_DIR = '../data/'
OUTPUT_DIR = '../out/'
def pre_process(bgr_img):
    """
    preprocess
    """
    #get img shape
    orig_shape = bgr_img.shape[:2]
    #resize img
    test_img = cv2.resize(bgr_img, (MODEL_WIDTH, MODEL_HEIGHT)).astype(np.float16)
    test_img = test_img.transpose([0, 3, 1, 2]).copy()
    
    # save memory C_CONTIGUOUS mode
    if not test_img.flags['C_CONTIGUOUS']:
        test_img = np.ascontiguousarray(gray_img)

    return orig_shape, test_img

def post_process(infer_output, image_file):
    print("post process")
    data = infer_output[0]
    
    img = np.reshape(data*255,(512,512))

    # Save the result
    resultimage=Image.fromarray(np.uint8(img))
    output_path = os.path.join(os.path.join(SRC_PATH, "../outputs"), os.path.basename(image_file))
    resultimage.save(output_path)
    print("result save success")    
    return 

def construct_image_info():
    """construct image info"""
    image_info = np.array([MODEL_WIDTH, MODEL_HEIGHT, 
                           MODEL_WIDTH, MODEL_HEIGHT], 
                           dtype = np.float32) 
    return image_info

def main():   
    """
    main
    """
    #create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    #acl init
    acl_resource = AclResource()
    acl_resource.init()

    #load model
    model = Model(MODEL_PATH)
    src_dir = os.listdir(INPUT_DIR)
    #infer picture
    for pic in src_dir:
        if not pic.lower().endswith(('.bmp', '.dib', '.png', '.jpg', 
                                    '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            print('it is not a picture, %s, ignore this file and continue,' % pic)
            continue
  
        #read picture
        pic_path = os.path.join(INPUT_DIR, pic)
        bgr_img = cv2.imread(pic_path)

        #get pic data
        orig_shape, test_img = pre_process(bgr_img)

        #inference
        result_list = model.execute([test_img, ])    

        #post_process
        post_process(result_list, pic)
if __name__ == '__main__':
    main()



