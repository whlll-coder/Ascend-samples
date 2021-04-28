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
*/

#ifndef MIC_H_
#define MIC_H_


extern "C" {
#include "driver/peripheral_api.h"
}


#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)


#define ASCEND_MIC_ERROR 0xFF
#define ASCEND_MIC_SUCCESS 0
class Mic {
public:
    Mic(void) ;
    ~Mic(void) ;

    int MicOpen(void);
    int MicClose(void);
    int MicCap(CAP_MIC_CALLBACK tfunc, void* param);
    int MicReadSound(void* pdata, int *size);
    int MicSetProperty(struct MICProperties *propties);
    int MicGetProperty(struct MICProperties *propties);
    int MicQryStatus();

private:
};

#endif // mic_H_
