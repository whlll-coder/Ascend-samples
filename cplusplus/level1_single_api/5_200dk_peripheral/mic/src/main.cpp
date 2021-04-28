/**
* @file Main.cpp
*
* Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <iostream>
#include <unistd.h>
#include <memory>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include "mic.h"

using namespace std;
unsigned int cnt = 0;
#define MAX_CAP_NUM  200
#define SLEEP_TIME  7000
FILE *file = nullptr ;
int mic_callback (const void* pdata, int size, void * param)
{
    fwrite(pdata, size, 1, file);
    cnt ++ ;
    INFO_LOG("mic_read_sound %d %d!",size, cnt);
    return 0;
}

int main(int argc, char* argv[])
{
    Mic micDevice;
    int  ret = micDevice.MicOpen();
    if(ASCEND_MIC_SUCCESS != ret )
    {
        ERROR_LOG("mic_open error");
        return 0;
    }

    MICProperties tproperty;
    tproperty.sample_rate = MIC_AUDIO_SAMPLE_RATE_44100;
    tproperty.frame_sample_rate = MIC_SAMPLE_NUM_1024;
    tproperty.cap_mode = MIC_CAP_PASSIVE ;
    tproperty.bit_width = MIC_AUDIO_BIT_WIDTH_16;
    tproperty.sound_mode = MIC_AUDIO_SOUND_MODE_STEREO;
    /* buf 44100X16/8 X1024/44100 *2 */

    ret = micDevice.MicSetProperty(&tproperty);
    if(ASCEND_MIC_SUCCESS != ret )
    {
        ERROR_LOG("mic_set_property error");
        return 0;
    }

    ret = micDevice.MicGetProperty(&tproperty);
    INFO_LOG("MicGetProperty sample_rate:%d frame_sample_rate:%d cap_mode:%d bit_width:%d sound_mode:%d",
    tproperty.sample_rate ,tproperty.frame_sample_rate,tproperty.cap_mode,tproperty.bit_width,tproperty.sound_mode);

    time_t timep;
    struct tm *p = nullptr;
    char name[256] = {0};

    time(&timep);
    p = localtime(&timep);

    sprintf(name, "./%d-%d-%d-%d-%02d.pcm",1900+p->tm_year,1+p->tm_mon,p->tm_mday,p->tm_hour,p->tm_min);
    file = fopen(name, "a+b");
    ret = micDevice.MicCap(mic_callback, nullptr);
    if(ASCEND_MIC_SUCCESS != ret )
    {
        ERROR_LOG("mic_cap error");
        micDevice.MicClose();
        fclose(file);
        return 0;
    }

    while(cnt < MAX_CAP_NUM )
    {
        usleep(SLEEP_TIME);
    }

    INFO_LOG("mic_read_sound end \n");
    micDevice.MicClose();
    fclose(file);
    return 0;

}