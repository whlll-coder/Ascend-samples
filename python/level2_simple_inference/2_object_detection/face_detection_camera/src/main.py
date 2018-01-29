import sys
sys.path.append("../../../../common")
sys.path.append("../")
project_path = sys.path[0] + "/../"
sys.path.append(project_path)
import datetime
import time
from sklearn.preprocessing import normalize
from scipy.linalg import norm
import numpy as np
np.seterr(divide='ignore',invalid='ignore')

from atlas_utils.camera import Camera
import atlas_utils.presenteragent.presenter_channel as presenter_channel
from atlas_utils.acl_model import Model
from atlas_utils.acl_resource import AclResource
from vgg_ssd import VggSsd

MODEL_PATH = project_path + "/model/face_detection.om"
MODEL_WIDTH = 304
MODEL_HEIGHT = 300
FACE_DETEC_CONF= project_path + "/scripts/face_detection.conf"
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720

def main():
    """main"""
    #Initialize acl
    acl_resource = AclResource()
    acl_resource.init()
    #Create a detection network instance, currently using the vgg_ssd network. 
    # When the detection network is replaced, instantiate a new network here
    detect = VggSsd(acl_resource, MODEL_WIDTH, MODEL_HEIGHT)
    #Load offline model
    model = Model(MODEL_PATH)
    #Connect to the presenter server according to the configuration, 
    # and end the execution of the application if the connection fails
    chan = presenter_channel.open_channel(FACE_DETEC_CONF)
    if chan is None:
        print("Open presenter channel failed")
        return
    #Open the CARAMER0 camera on the development board
    cap = Camera(0)
    #人脸注册
    while True:
        #从摄像头依次读取一张读取一张人脸进行注册
        print("请无关人员撤离 准备进行人脸注册")
        print("请将摄像头对准自己 听到哔声后查看注册结果")
        print("\n5s 后开始注册\n") 
        time.sleep(5)
        print("======================================")
       
        #Read a picture from the camera
        image_register = cap.read()
        if image_register is None:
            print("Get memory from camera failed")
            break

        #The detection network processes images into model input data
        model_input_register = detect.pre_process(image_register)
        if model_input_register is None:
            print("Pre process image failed")
            break

        #Send data to offline model inference
        result_register = model.execute(model_input_register)
        box_info_register = result_register[1][0]
        score_register = box_info_register[0, 2]
        if score_register >0.90:
            print("参数提取成功 识别率: " + str(score_register))
            name = input("请输入您的名字: ")
        else:
            print("参数提取失败 识别率: " + str(score_register) + "正在进行重新匹配...")
            continue

        if name == "":
            print("请告诉我你的名字哦 ~\n\n")
            continue
        if name is not None:
            print("匹配成功 下面进行人脸识别")
            break
    #flag是一个小标记 用来控制 while 循环
    flag = 0
    #注册人脸的模型输出结果以及识别率
    box_info_register = result_register[1][0]
    score_register = box_info_register[0, 2]
    # 5 秒后开始进行人脸检测
    print("\n5s 后开始进行人脸检测\n")
    print("======================================")
    time.sleep(5)

    while True:
        #Read a picture from the camera
        image = cap.read()
        if image is None:
            print("Get memory from camera failed")
            break
        
        #The detection network processes images into model input data
        model_input = detect.pre_process(image)
        if model_input is None:
            print("Pre process image failed")
            break
        #Send data to offline model inference
        result = model.execute(model_input)
        #Detecting network analysis inference output
        jpeg_image, detection_list = detect.post_process(result, image)
        if jpeg_image is None:
            print("The jpeg image for present is None")
            break
        
        chan.send_detection_data(CAMERA_FRAME_WIDTH, CAMERA_FRAME_HEIGHT, 
                                 jpeg_image, detection_list)
        #余弦距离可适应用于人脸识别，将待识别人脸的图像提取特征，与人脸注册库的所有图像的特征矩阵求距离，然后找到最相似的
        box_info = result[1][0]
        score = box_info[0, 2]
        box_info_register = box_info_register[~np.isnan(box_info_register)]
        box_info = box_info[~np.isnan(box_info)]

        norm1 = norm(box_info,axis=-1)
        norm2 = norm(box_info_register,axis=-1)
        end_norm = np.dot(norm1,norm2)
        cos = np.dot(box_info, box_info_register.T)/end_norm

        similarity= 0.5*cos+0.5

        similarity = similarity[~np.isnan(similarity)]

        if similarity > 0.93 and score > 0.90 and flag == 0:
            print(cos.shape)
            print("Hi 你好呀 "+name)
            print(np.all(similarity>0.75))
            k=0
            for i in np.nditer(similarity, order='C'):
                k = k + 1
                print(k)
            print("cos:\n"+str(cos)+"\n"+"similarity:\n"+str(similarity))
            print(type(box_info_register))
            print("box_info_register:\n"+str(box_info_register)+"\n"+"box_info:\n"+str(box_info))
            print("score_register: "+str(score_register)+"\n"+"score: "+str(score))
        if similarity < 0.93 and score > 0.90 and flag == 0:
            print("未注册人脸信息 你好 陌生人")
            print(np.all(similarity<0.68))
            print(type(similarity))
            print("cos:\n"+str(cos)+"\n"+"similarity:\n"+str(similarity))
        if score < 0.90:
            print("未能成功检测人脸 识别率: "+str(score))
            continue
        flag = 1


if __name__ == '__main__':
    main()
