# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import asyncore
import numpy as np
import pickle
import socket
import struct
import cv2
import time
import pyrealsense2 as rs
import pydobot

from math import cos, sin, fabs, atan, pi, sqrt, acos
from serial.tools import list_ports

# Dobot config
dobot_port = list_ports.comports()[0].device
dobot_device = pydobot.Dobot(port=dobot_port, verbose=True)

# RealSense config
align_to = rs.stream.color
align = rs.align(align_to)

# color for show detection results
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

# RealSense D435 Ethernet config
print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
MC_IP_ADDRESS = '192.168.8.134'  # 200DK ip
PORT = 1024
CHUNK_SIZE = 4096

# for robot control: move robot every 5 frames
FRAME_COUNT = 0
ROBOT_MOVE_FREQUENCY = 5

PI = 180  # math pi=PI


class DevNullHandler(asyncore.dispatcher_with_send):
    """
    Handle null receive
    """

    def handle_read(self):
        """
        Read data and print
        :return: null
        """

        print(self.recv(1024))

    def handle_close(self):
        """
        Close connection
        :return: null
        """

        self.close()


class EtherSenseServer(asyncore.dispatcher):
    """
    UDP client for each camera
    """

    def __init__(self, address):
        asyncore.dispatcher.__init__(self)
        print("Launching Realsense Camera Server")
        try:
            self.pipeline, self.sensor = open_pipeline()
        except:
            print("Unexpected error: ", sys.exc_info()[1])
            sys.exit(1)
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        print('sending acknowledgement to', address)

        # reduce the resolution of the depth image using post processing
        self.decimate_filter = rs.decimation_filter()
        self.decimate_filter.set_option(rs.option.filter_magnitude, 2)
        self.frame_data = ''
        self.connect((MC_IP_ADDRESS, PORT))
        self.packet_id = 0

        self.remaining_bytes = 0
        self.buffer = bytearray()

        self.color_image = np.array([])
        self.depth_image = np.array([])
        self.depth_scale = 0
        self.aligned_frames = []

        # yolov3 detection result
        boxes = dict()
        boxes['detection_classes'] = []
        boxes['detection_boxes'] = []
        boxes['detection_scores'] = []
        boxes['bgr_img'] = []
        self.boxes = boxes

    def handle_connect(self):
        """
        Print UDP connection messages
        :return: null
        """

        print("connection received")

    def writable(self):
        """
        Want write notifies
        :return: True
        """

        return True

    def update_frame(self):
        """
        Read image data from RealSense
        :return: null
        """

        # get image from RealSense
        color_mat, depth_mat, timestamp, aligned_depth_frame = get_image_and_timestamp(self.pipeline)
        if color_mat is not None:
            # convert the depth image to a string for broadcast
            data = pickle.dumps(color_mat)
            # capture the lenght of the data portion of the message
            length = struct.pack('<I', len(data))
            # include the current timestamp for the frame
            ts = struct.pack('<d', timestamp)
            # for the message for transmission
            self.frame_data = b''.join([length, ts, data])

            self.color_image = color_mat
            self.depth_image = depth_mat
            self.depth_scale = self.sensor.get_depth_scale()
            self.aligned_frames = aligned_depth_frame
            self.depth_intrinsics = rs.video_stream_profile(aligned_depth_frame.profile).get_intrinsics()

    def handle_write(self):
        """
        Send image data to 200DK
        :return: null
        """

        # first time the handle_write is called
        if not hasattr(self, 'frame_data'):
            self.update_frame()
        # the frame has been sent in it entirety so get the latest frame
        if len(self.frame_data) == 0:
            self.update_frame()
        else:
            # send the remainder of the frame_data until there is no data remaining for transmition
            remaining_size = self.send(self.frame_data)
            self.frame_data = self.frame_data[remaining_size:]

    def handle_read(self):
        """
        Read inference result from 200DK
        :return: null
        """

        if self.remaining_bytes == 0:
            # get the expected frame size
            self.frame_length = struct.unpack('<I', self.recv(4))[0]
            # get the timestamp of the current frame
            self.remaining_bytes = self.frame_length

        # request the frame data until the frame is completely in buffer
        data = self.recv(self.remaining_bytes)
        self.buffer += data
        self.remaining_bytes -= len(data)
        # once the frame is fully received, process/display it
        # get 200DK inference result
        if len(self.buffer) == self.frame_length:
            inference_result = pickle.loads(self.buffer)
            self.buffer = bytearray()
            self.boxes = inference_result

        # legal result, show result and move robot
        if self.color_image.shape[0] > 0:
            cv2.namedWindow('Ascend Example', cv2.WINDOW_NORMAL)
            result_image = self.color_image.copy()

            if len(self.boxes['detection_boxes']) > 0:
                result_image = show_reuslt(result_image, self.boxes)

                # robot move: move robot every FRAME_COUNT frames
                global FRAME_COUNT
                FRAME_COUNT = FRAME_COUNT + 1
                if FRAME_COUNT % 5 == 0:
                    move_robot(self.depth_intrinsics, self.depth_scale, self.depth_image, self.boxes)

            cv2.imshow('Ascend Example', result_image)
            key = cv2.waitKey(1)

    def handle_close(self):
        """
        Close connection
        :return: null
        """

        self.close()


def get_image_and_timestamp(pipeline):
    """
    Read image data and timestamp from RealSense
    :param pipeline: RealSense pipeline
    :return: color and depth image and timestamp
    """

    frames = pipeline.wait_for_frames()
    # take owner ship of the frame for further processing
    frames.keep()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if depth_frame and color_frame:
        # take owner ship of the frame for further processing
        # color_frame.keep()
        # represent the frame as a numpy array
        color_mat = np.asanyarray(color_frame.get_data())

        # depth_frame.keep()
        # represent the frame as a numpy array
        depth_mat = np.asanyarray(depth_frame.get_data())

        ts = frames.get_timestamp()
        return color_mat, depth_mat, ts, depth_frame

    else:
        return None, None, None, None


def open_pipeline():
    """
    Start RealSense pipeline
    :return: realSense pipeline and depth_sensor
    """

    # RealSense start
    frame_width = 640
    frame_height = 480
    fps = 30
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, fps)
    pipeline = rs.pipeline()
    pipeline_profile = pipeline.start(cfg)
    sensor = pipeline_profile.get_device().first_depth_sensor()

    return pipeline, sensor


def show_reuslt(image, boxes):
    """
    Show object detection result
    :param image: original image
    :param boxes: object detection result
    :return: image with boxes
    """

    top, left, bottom, right = boxes['detection_boxes'][0]  # y_min, x_min, y_max, x_max
    top = np.clip(top, 1, image.shape[0]).astype(np.int)
    left = np.clip(left, 1, image.shape[1]).astype(np.int)
    bottom = np.clip(bottom, 1, image.shape[0]).astype(np.int)
    right = np.clip(right, 1, image.shape[1]).astype(np.int)
    confidence = boxes['detection_scores'][0]

    cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), COLORS[0], 2)
    p3 = (max(int(left), 15), max(int(top) - 5, 15))
    cv2.putText(image, boxes['detection_classes'][0] + " " + str(confidence), p3, cv2.FONT_ITALIC, 0.4, COLORS[0], 1)

    return image


def pixel_to_point(intrin, pixel, depth):
    """
    Compute the 3D camera coordinates from pixel coordinates and depth in an image
    :param intrin: camera intrin
    :param pixel: [x, y] of picture coordinate
    :param depth: depth of pixel
    :return: [x, y, z] of camera 3D coordinate
    """

    x = (pixel[0] - intrin.ppx) / intrin.fx
    y = (pixel[1] - intrin.ppy) / intrin.fy
    r2 = x * x + y * y
    f = 1 + intrin.coeffs[0] * r2 + intrin.coeffs[1] * r2 * r2 + intrin.coeffs[4] * r2 * r2 * r2
    ux = x * f + 2 * intrin.coeffs[2] * x * y + intrin.coeffs[3] * (r2 + 2 * x * x)
    uy = y * f + 2 * intrin.coeffs[3] * x * y + intrin.coeffs[2] * (r2 + 2 * y * y)
    x = ux
    y = uy

    # meters to mm
    m_to_mm = 1000
    point = [0, 0, 0]
    point[0] = m_to_mm * depth * x  # camera 3D coordinate x, same as robot coordinate -z
    point[1] = m_to_mm * depth * y  # camera 3D coordinate y, same as robot coordinate -y
    point[2] = m_to_mm * depth  # camera 3D coordinate z, same as robot coordinate x

    return point


def point_to_point(extrin, from_point):
    """
    Transform 3D coordinates with camera extrin
    :param extrin: camera extrin: rotation and translation
    :param from_point: point in original 3D coordinates
    :return: point in new 3D coordinates
    """

    to_point = [0, 0, 0]
    to_point[0] = extrin["rotation"][0] * from_point[0] + extrin["rotation"][3] * from_point[1] + extrin["rotation"][
        6] * from_point[2] + extrin["translation"][0]
    to_point[1] = extrin["rotation"][1] * from_point[0] + extrin["rotation"][4] * from_point[1] + extrin["rotation"][
        7] * from_point[2] + extrin["translation"][1]
    to_point[2] = extrin["rotation"][2] * from_point[0] + extrin["rotation"][5] * from_point[1] + extrin["rotation"][
        8] * from_point[2] + extrin["translation"][2]

    return to_point


def get_extrin(translation_x, translation_y, translation_z, roll, pitch, yaw):
    """
    Get extrin of camera
    :param translation_x: translation in x-axis
    :param translation_y: translation in y-axis
    :param translation_z: translation in z-axis
    :param roll: rotation in x-axis
    :param pitch: rotation in y-axis
    :param yaw: rotation in z-axis
    :return: extrin matrix
    """

    extrin = {}
    rotation_x = np.array([[1, 0, 0],
                           [0, cos(roll), -sin(roll)],
                           [0, sin(roll), cos(roll)]])
    rotation_y = np.array([[cos(pitch), 0, sin(pitch)],
                           [0, 1, 0],
                           [-sin(pitch), 0, cos(pitch)]])
    rotation_z = np.array([[cos(yaw), -sin(yaw), 0],
                           [sin(yaw), cos(yaw), 0],
                           [0, 0, 1]])

    rotation = rotation_x * rotation_y * rotation_z
    extrin["rotation"] = rotation.reshape(-1, 9).tolist()[0]
    extrin["translation"] = [translation_x, translation_y, translation_z]

    return extrin


def calculate_robot_action(cur_world_coordinate, dest_world_coordinate, dest_camera_coordinate, cur_j2, cur_j3):
    """
    Calculate robot's j1、j2、j3 move angle using ascend_logo's current and target 3D point
    :param cur_world_coordinate: ascend_logo's current world 3D coordinate
    :param dest_world_coordinate: ascend_logo's target world 3D coordinate
    :param dest_camera_coordinate: ascend_logo's target camera 3D coordinate
    :param cur_j2: robot's current j2 angle
    :param cur_j3: robot's current j3 angle
    :return: robot's j1、j2、j3 move angle increment
    """

    robot_action = [0, 0, 0]  # [J1 angle increment, J2 angle increment, J3 angle increment]
    cur_pos = np.array(cur_world_coordinate)
    dest_pos = np.array(dest_world_coordinate)

    # robot arm length
    j3_len = 147
    j2_len = 135

    # J1 move
    robot_action[0] = np.arccos(
        cur_pos.dot(dest_pos) / (sqrt(cur_pos.dot(cur_pos)) * sqrt(dest_pos.dot(dest_pos)))) * PI / pi
    if dest_pos[1] > cur_pos[1]:
        robot_action[0] = -robot_action[0]

    # J2 J3 move
    target_distance = 300  # mm, ideal distance between RealSense camera and ascend logo
    x_range = 70  # mm, robot's move range of x-axis

    # increment of x-axis and z-axis
    deta_x = (dest_camera_coordinate[2] - target_distance) / target_distance * x_range
    deta_z = dest_world_coordinate[2] - cur_world_coordinate[2]

    # calculate j3 angle increment theta3 first
    alpha_2 = pi / 2 - cur_j2 * pi / PI
    alpha_3 = -cur_j3 * pi / PI
    r_x = deta_x + j2_len * cos(alpha_2) + j3_len * cos(alpha_3)
    r_z = deta_z + j2_len * sin(alpha_2) + j3_len * sin(alpha_3)

    # clip cos result to [-1, 1]
    cos_tmp = (j3_len ** 2 - j2_len ** 2 + r_x ** 2 + r_z ** 2) / (2 * j3_len * sqrt(r_x ** 2 + r_z ** 2))
    cos_tmp = np.clip(cos_tmp, -1, 1)
    robot_action[2] = atan(r_z / r_x) - acos(cos_tmp) - alpha_3
    theta3 = robot_action[2] * PI / pi

    # calculate j2 angle increment theta2
    # clip cos result to [-1, 1]
    cos_tmp = (r_x - j3_len * cos(alpha_3 + robot_action[2])) / j2_len
    cos_tmp = np.clip(cos_tmp, -1, 1)
    robot_action[1] = acos(cos_tmp) - alpha_2
    theta2 = robot_action[1] * PI / pi

    robot_action[1] = -theta2
    robot_action[2] = theta3

    return robot_action


def move_robot(depth_intrinsics, depth_scale, depth_image, boxes):
    """
    Calculate robot j1、j2、j3 move angle increment and move it
    :param depth_intrinsics: intrinsics of RealSense camera
    :param depth_scale: scale of pixel value and true depth
    :param depth_image: depth image of RealSense camera
    :param boxes: yolov3 detection result
    :return: None
    """

    # yolov3 detection boxes
    top, left, bottom, right = boxes['detection_boxes'][0]
    top = np.clip(top, 1, depth_image.shape[0]).astype(np.int)
    left = np.clip(left, 1, depth_image.shape[1]).astype(np.int)
    bottom = np.clip(bottom, 1, depth_image.shape[0]).astype(np.int)
    right = np.clip(right, 1, depth_image.shape[1]).astype(np.int)

    # robot's current state
    (x, y, z, r, j1, j2, j3, j4) = dobot_device.pose()
    # print(f'x:{x} y:{y} z:{z} j1:{j1} j2:{j2} j3:{j3} j4:{j4}')

    # calculate ascend_logo's camera and world 3D coordinate
    roi_width = 10  # calculate distance using a (2 * roi_width) *　(2 * roi_width)　area
    ascend_distance = np.mean(depth_image[int((top + bottom) / 2) - roi_width:int((top + bottom) / 2) + roi_width,
                              int((left + right) / 2) - roi_width:int((left + right) / 2) + roi_width]) * depth_scale
    ascend_center_point = [(top + bottom) / 2, (left + right) / 2]
    print("ascend_center_point:pixel", ascend_center_point, "  ascend_distance/m:", ascend_distance)
    ascend_camera_3d_point = pixel_to_point(depth_intrinsics, ascend_center_point, ascend_distance)
    print("ascend_camera_3D_point xyz/mm:", ascend_camera_3d_point)
    extrin = get_extrin(x, y, z, 0, 0, j1 / PI * pi)
    ascend_world_3d_point = point_to_point(extrin, from_point=[ascend_camera_3d_point[2],
                                                               -ascend_camera_3d_point[1],
                                                               -ascend_camera_3d_point[0]])
    print("ascend_world_3D_point xyz/mm:", ascend_world_3d_point)

    # calculate picture_center's camera and world 3D coordinate
    height, width = depth_image.shape
    target_distance = ascend_distance  # using ascend_distance directly
    target_center_point = [height / 2, width / 2 + 50]  # 50mm for color camera position
    print("target_center_point:pixel", target_center_point, "  target_distance/m:", target_distance)
    target_camera_3d_point = pixel_to_point(depth_intrinsics, target_center_point, ascend_distance)
    print("target_camera_3D_point xyz/mm:", target_camera_3d_point)
    target_world_3d_point = point_to_point(extrin, from_point=[target_camera_3d_point[2],
                                                               -target_camera_3d_point[1],
                                                               -target_camera_3d_point[0]])
    print("target_world_3D_point xyz/mm:", target_world_3d_point)

    # calculate j1 j2 J3 move angle increment
    robot_action = calculate_robot_action(cur_world_coordinate=ascend_world_3d_point,
                                          dest_world_coordinate=target_world_3d_point,
                                          dest_camera_coordinate=target_camera_3d_point,
                                          cur_j2=j2,
                                          cur_j3=j3)
    print("robot's j1、j2、j3 move angle increment:", robot_action)

    # j1 j2 J3 move when increment > thresh
    j1_thresh = 6
    j2_thresh = 5
    j3_thresh = 6

    # limit angle of j1 j2 J3
    j1_limit_min = -80
    j1_limit_max = 80
    j2_limit_min = 5
    j2_limit_max = 60
    j3_limit_min = 5
    j3_limit_max = 70

    # move when flag is True
    flag_j1 = abs(robot_action[0]) > j1_thresh
    flag_j2 = abs(robot_action[1]) > j2_thresh
    flag_j3 = abs(robot_action[2]) > j3_thresh

    # move start
    step = 3  # use step to move a little once time
    if flag_j1 and (not flag_j2) and (not flag_j3):  # j1 move
        new_j1 = j1 + robot_action[0] / step
        dobot_device.move_by_angle(np.clip(new_j1, j1_limit_min, j1_limit_max),
                                   np.clip(j2, j2_limit_min, j2_limit_max),
                                   np.clip(j3, j3_limit_min, j3_limit_max),
                                   j4, wait=False)
    elif flag_j2 and (not flag_j1) and (not flag_j3):  # j2 move
        new_j2 = j2 + robot_action[1] / step
        dobot_device.move_by_angle(np.clip(j1, j1_limit_min, j1_limit_max),
                                   np.clip(new_j2, j2_limit_min, j2_limit_max),
                                   np.clip(j3, j3_limit_min, j3_limit_max),
                                   j4, wait=False)
    elif flag_j3 and (not flag_j1) and (not flag_j2):  # j3 move
        new_j3 = j3 + robot_action[2] / step
        dobot_device.move_by_angle(np.clip(j1, j1_limit_min, j1_limit_max),
                                   np.clip(j2, j2_limit_min, j2_limit_max),
                                   np.clip(new_j3, j3_limit_min, j3_limit_max),
                                   j4, wait=False)
    elif flag_j1 and flag_j2 and (not flag_j3):  # j1 and j2 move
        new_j1 = j1 + robot_action[0] / step
        new_j2 = j2 + robot_action[1] / step
        dobot_device.move_by_angle(np.clip(new_j1, j1_limit_min, j1_limit_max),
                                   np.clip(new_j2, j2_limit_min, j2_limit_max),
                                   np.clip(j3, j3_limit_min, j3_limit_max),
                                   j4, wait=False)
    elif flag_j1 and flag_j3 and (not flag_j2):  # j1 and j3 move
        new_j1 = j1 + robot_action[0] / step
        new_j3 = j3 + robot_action[2] / step
        dobot_device.move_by_angle(np.clip(new_j1, j1_limit_min, j1_limit_max),
                                   np.clip(j2, j2_limit_min, j2_limit_max),
                                   np.clip(new_j3, j3_limit_min, j3_limit_max),
                                   j4, wait=False)
    elif flag_j2 and flag_j3 and (not flag_j1):  # j2 and j3 move
        new_j2 = j2 + robot_action[1] / step
        new_j3 = j3 + robot_action[2] / step
        dobot_device.move_by_angle(np.clip(j1, j1_limit_min, j1_limit_max),
                                   np.clip(new_j2, j2_limit_min, j2_limit_max),
                                   np.clip(new_j3, j3_limit_min, j3_limit_max),
                                   j4, wait=False)
    elif flag_j1 and flag_j3 and flag_j2:  # j1 and j2 and j3 move
        new_j1 = j1 + robot_action[0] / step
        new_j2 = j2 + robot_action[1] / step
        new_j3 = j3 + robot_action[2] / step
        dobot_device.move_by_angle(np.clip(new_j1, j1_limit_min, j1_limit_max),
                                   np.clip(new_j2, j2_limit_min, j2_limit_max),
                                   np.clip(new_j3, j3_limit_min, j3_limit_max),
                                   j4, wait=False)
    else:
        pass


def main(argv):
    """
    Main loop function
    :param argv: none
    :return: null
    """
    # initalise the EtherSenseServer
    server = EtherSenseServer(MC_IP_ADDRESS)
    asyncore.loop()


if __name__ == '__main__':
    main(sys.argv[1:])
