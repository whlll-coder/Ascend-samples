import datetime
import time
import acl
import configparser
import sys
import os

cur_file_dir = os.path.dirname(os.path.abspath(__file__))
 
sys.path.append(cur_file_dir + "/../../../../common")

import atlas_utils.video as video
from atlas_utils.constants import *
from atlas_utils.utils import *
from atlas_utils.acl_resource import AclResource
from atlas_utils.acl_image import AclImage
from atlas_utils.acl_logger import log_error, log_info

from yolov3 import Yolov3
from preprocess import Preprocess
from postprocess import DetectData, Postprocess


MODEL_PATH = "../model/yolov3_yuv.om"
MODEL_WIDTH = 416
MODEL_HEIGHT = 416
COCO_DETEC_CONF="../scripts/coco_detection.conf"

def create_threads(detector):
    config = configparser.ConfigParser()
    config.read(COCO_DETEC_CONF)
    video_decoders = []
    for item in config['videostream']:
        preprocesser = Preprocess(config['videostream'][item], 
                                  len(video_decoders),
                                  MODEL_WIDTH, MODEL_HEIGHT)
        video_decoders.append(preprocesser)

    rtsp_num = len(video_decoders)
    if rtsp_num == 0:
        log_error("No video stream name or addr configuration in ",
                  COCO_DETEC_CONF)
        return None, None

    postprocessor = Postprocess(detector)

    display_channel = int(config['display']['channel'])
    if (display_channel is None) or (display_channel >= rtsp_num):
        log_info("No video to display, display configuration: ", 
                 config['display']['channel'])
    else:
        video_decoders[display_channel].set_display(True)
        
    return video_decoders, postprocessor


def main():
    """
    Function description:
        Main function
    """
    acl_resource = AclResource()
    acl_resource.init()   

    detector = Yolov3(acl_resource, MODEL_PATH, MODEL_WIDTH, MODEL_HEIGHT)

    video_decoders, postprocessor = create_threads(detector)
    if video_decoders is None:
        log_error("Please check the configuration in %s is valid"
                  %(COCO_DETEC_CONF))
        return
    
    while True:
        all_process_fin = True
        for decoder in video_decoders:
            ret, data = decoder.get_data()
            if ret == False:                
                continue
            if data:
                detect_results = detector.execute(data)
                postprocessor.process(data, detect_results)
                
            all_process_fin = False
        if all_process_fin:
            log_info("all video decoder finish")
            break

    postprocessor.exit()

    log_info("sample execute end")  


if __name__ == '__main__':
    main()
