/**
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

* File main.cpp
* Description: dvpp sample main func
*/

#include <iostream>
#include <stdlib.h>
#include <dirent.h>

#include "super_resolution_process.h"
#include "utils.h"
using namespace std;

namespace {
}

int main(int argc, char *argv[]) {
    //检查应用程序执行时的输入,程序执行要求输入图片目录参数
    if((argc < 2) || (argv[1] == nullptr)){
        ERROR_LOG("Please input: ./main <image_dir>");
        return FAILED;
    }

    // model type 
    // 0 - SRCNN
    // 1 - FSRCNN
    // 2 - ESPCN
    // (defalt: SRCNN)
    uint8_t kModelType;
    if (argv[2] == nullptr) kModelType = 0;
    else kModelType = atoi(argv[2]);

    //实例化超分辨率推理对象
    SuperResolutionProcess SR(kModelType);
    //初始化acl资源
    Result ret = SR.Init();
    if (ret != SUCCESS) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    //获取图片目录下所有的图片文件名
    string inputImageDir = string(argv[1]);
    vector<string> fileVec;
    Utils::GetAllFiles(inputImageDir, fileVec);
    if (fileVec.empty()) {
        ERROR_LOG("Failed to deal all empty path=%s.", inputImageDir.c_str());
        return FAILED;
    }
    //逐张图片推理
    for (string imageFile : fileVec) {
        //预处理图片,初始化模型和内存
        ret = SR.Preprocess(imageFile);
        if (ret != SUCCESS) {
            ERROR_LOG("Deal file %s failed, continue to read next",
                      imageFile.c_str());                
            continue;
        }
        //将预处理的图片送入模型推理,并获取推理结果
        aclmdlDataset* inferenceOutput = nullptr;
        ret = SR.Inference(inferenceOutput);
        if ((ret != SUCCESS) || (inferenceOutput == nullptr)) {
            ERROR_LOG("Inference model inference output data failed");
            return FAILED;
        }
        //解析推理输出
        ret = SR.Postprocess(imageFile, inferenceOutput);
        if (ret != SUCCESS) {
            ERROR_LOG("Process model inference output data failed");
            return FAILED;
        }
        // Destroy model
        SR.DestroyModel();
    }

    INFO_LOG("Execute sample success");
    return SUCCESS;
}
