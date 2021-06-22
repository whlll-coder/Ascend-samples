/**
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#pragma once
#include <iostream>
#include <unistd.h>
#include <dirent.h>
#include <fstream>
#include <cstring>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <map>

#include <stdint.h>

#include <string>
#include <memory>
#include <thread>
#include "atlasutil/atlas_utils.h"
#include "atlasutil/atlas_error.h"
#include "atlasutil/acl_device.h"

#define INVALID_STREAM_FORMAT -1

#define RTSP_TRANSPORT_UDP "udp"
#define RTSP_TRANSPORT_TCP "tcp"

#define VIDEO_CHANNEL_MAX  23
#define INVALID_CHANNEL_ID -1

enum StreamType {
    STREAM_VIDEO = 0,
    STREAM_RTSP,
};

enum DecodeStatus {
    DECODE_ERROR  = -1,
    DECODE_UNINIT = 0,
    DECODE_READY  = 1,
    DECODE_START  = 2,
    DECODE_FFMPEG_FINISHED = 3,
    DECODE_DVPP_FINISHED = 4,
    DECODE_FINISHED = 5
};


typedef struct PicDesc {
    std::string picName;
    int width;
    int height;
}PicDesc;

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
}Result;


