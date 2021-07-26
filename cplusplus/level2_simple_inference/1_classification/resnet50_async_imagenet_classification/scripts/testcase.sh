
caffe_model="https://obs-model-ascend.obs.cn-east-2.myhuaweicloud.com/resnet50/resnet50.caffemodel"
caffe_prototxt="https://obs-model-ascend.obs.cn-east-2.myhuaweicloud.com/resnet50/resnet50.prototxt"
data_source="https://c7xcode.obs.cn-north-4.myhuaweicloud.com/models/aclsample/"
model_name="resnet50"
project_name="cplusplus_resnet50_async_imagenet_classification"
data_name="dog1_1024_683.bin" 
verify_source="https://c7xcode.obs.cn-north-4.myhuaweicloud.com/models/aclsample/"
verify_name="result.txt"

version=$1

script_path="$( cd "$(dirname $BASH_SOURCE)" ; pwd -P)"
project_path=${script_path}/..
common_script_dir=${script_path}/../../../../../common/

run_command="./main "
model_atc="atc --model=${project_path}/caffe_model/${caffe_prototxt##*/} --weight=${project_path}/caffe_model/${caffe_model##*/} --framework=0 --output=${HOME}/models/${project_name}/${model_name} --soc_version=Ascend310 --input_format=NCHW --input_fp16_nodes=data --output_type=FP32 --out_nodes=prob:0
"

declare -i success=0
declare -i inferenceError=1
declare -i verifyResError=2

. ${common_script_dir}/testcase_common.sh

function main() {

    # 下载测试集和验证集
    downloadDataWithVerifySource
    if [ $? -ne 0 ];then
        return ${inferenceError}
    fi

    # 转模型
    modelconvert
    if [ $? -ne 0 ];then
        return ${inferenceError}
    fi   

    buildproject
    if [ $? -ne 0 ];then
        return ${inferenceError}
    fi

    run_picture
    if [ $? -eq ${inferenceError} ];then
        return ${inferenceError}
    elif [ $? -eq ${verifyResError} ];then
	return ${verifyResError}
    fi

    return ${success}
}
main