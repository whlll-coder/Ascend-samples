#!/bin/bash
common_script_dir=`pwd`

function downloadDataWithVerifySource()
{
  if [[ ${version}"x" = "x" ]];then
    echo "ERROR: version is invalid"
    return ${inferenceError}
  fi
  mkdir -p ${project_path}/data/
  if [[ $(find ${project_path}/data -name ${data_name})"x" = "x" ]];then
    wget -O ${project_path}/data/${data_name}  ${data_source}${data_name}  --no-check-certificate
    if [ $? -ne 0 ];then
      echo "download test data failed, please check Network."
      return ${inferenceError}
    fi
  fi

  mkdir -p ${project_path}/verify_image/
  if [[ $(find ${project_path}/verify_image -name ${verify_name})"x" = "x" ]];then
    wget -O ${project_path}/verify_image/${verify_name} ${verify_source}${verify_name} --no-check-certificate
    if [ $? -ne 0 ];then
      echo "download verify data failed, please check Network."
      return ${inferenceError}
    fi
  fi
  
  return ${success}

}
function setAtcEnv() {
  # 设置模型转换时需要的环境变量
  export install_path=$HOME/Ascend/ascend-toolkit/latest
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:${LD_LIBRARY_PATH}

  return ${success}
}
function downloadOriginalModel() {

    mkdir -p ${project_path}/model/

    if [[ ${caffe_prototxt}"x" != "x" ]] && [[ ${caffe_model}"x" != "x" ]];then
      wget -O ${project_path}/model/${caffe_prototxt##*/} ${caffe_prototxt} --no-check-certificate
      if [ $? -ne 0 ];then
        echo "install caffe_prototxt failed, please check Network."
        return ${inferenceError}
      fi

      wget -O ${project_path}/model/${caffe_model##*/} ${caffe_model} --no-check-certificate
      if [ $? -ne 0 ];then
        echo "install caffe_model failed, please check Network."
        return ${inferenceError}
      fi
    elif [[ ${tf_model}"x" != "x" ]];then
      wget -O ${project_path}/model/${tf_model##*/} ${tf_model} --no-check-certificate
      if [ $? -ne 0 ];then
        echo "install caffe_prototxt failed, please check Network."
        return ${inferenceError}
      fi
    else
      echo "No model download link available, please confirm"
      return ${inferenceError}
    fi

    return ${success}
}
function modelconvert()
{
  mkdir -p ${HOME}/models/${project_name}
  if [[ $(find ${HOME}/models/${project_name} -name ${model_name}".om")"x" = "x" ]];then
    # 下载原始模型文件[aipp_cfg文件]
    downloadOriginalModel
    if [ $? -ne 0 ];then
      return ${inferenceError}
    fi

    # 设置模型转换的环境变量
    setAtcEnv

    # 转模型
    cd ${project_path}/model/
    ${model_atc}
    if [ $? -ne 0 ];then
      echo "ERROR: convert model failed"
      return ${inferenceError}
    fi
  fi

  if [[ $(find ${project_path}/model -name ${model_name}".om")"x" != "x" ]];then
    rm ${project_path}/model/${model_name}".om"
  fi

  ln -s ${HOME}/models/${project_name}/${model_name}".om" ${project_path}/model/${model_name}".om"
  if [ $? -ne 0 ];then
    echo "ERROR: failed to set model soft connection"
    return ${inferenceError}
  fi
  return ${success}

}
function buildproject()
{
  Kernel=`uname -m`
  if [[ ${Kernel} = "x86_64" ]];then
    TargetKernel="x86"
    cxx_compiler="g++"
    export DDK_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/x86_64-linux
  else
    TargetKernel="arm"
    cxx_compiler="aarch64-linux-gnu-g++"
    export DDK_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/arm64-linux
  fi
  export NPU_HOST_LIB=${DDK_PATH}/acllib/lib64/stub

  echo "cxx_compiler=${cxx_compiler}"
  # 创建目录用于存放编译文件
    mkdir -p ${project_path}/build/intermediates/host
    if [ $? -ne 0 ];then
        echo "ERROR: mkdir build folder failed. please check your project"
        return ${inferenceError}
    fi
    cd ${project_path}/build/intermediates/host


    # 产生Makefile
    cmake ${project_path}/src -DCMAKE_CXX_COMPILER=${cxx_compiler} -DCMAKE_SKIP_RPATH=TRUE
    if [ $? -ne 0 ];then
        echo "ERROR: cmake failed. please check your project"
        return ${inferenceError}
    fi

    make
    if [ $? -ne 0 ];then
        echo "ERROR: make failed. please check your project"
        return ${inferenceError}
    fi
    return ${success}
}
function run_picture()
{
    # 重新配置程序运行所需的环境变量
    export LD_LIBRARY_PATH=
    export LD_LIBRARY_PATH=/home/HwHiAiUser/Ascend/acllib/lib64:/home/HwHiAiUser/ascend_ddk/${TargetKernel}/lib:${LD_LIBRARY_PATH}
    
    mkdir -p ${project_path}/out/output
    # 运行程序
    cd ${project_path}/out
    ${run_command}
    if [ $? -ne 0 ];then
        echo "ERROR: run failed. please check your project"
        return ${inferenceError}
    fi

    # 调用python脚本判断本工程推理结果是否正常
    for outimage in $(find ${project_path}/verify_image -name "*.jpg");do
        tmp=`basename $outimage`
        if [[ ! -d "${project_path}/out/output" ]];then
            echo "ERROR: not find results folders!"
            return ${verifyResError}
        fi
        for test_file in `find ${project_path}/out/output -name "*${tmp#*_}"`;do
            python3.6 ${common_script_dir}/verify_result.py ${test_file} ${outimage}
            if [ $? -ne 0 ];then
                echo "ERROR: The result of reasoning is wrong!"
                return ${verifyResError}
            fi
        done
    done

    echo "run success"

    return ${success}
}
