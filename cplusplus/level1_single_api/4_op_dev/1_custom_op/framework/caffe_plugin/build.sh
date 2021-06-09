#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

###### Environment variable settings, need to set according to your own device ######

# CPLUS_INCLUDE_PATH: The include directory under the ProtoBuf installation path. If ProtoBuf is not specified with --prefix
#                     and the default configuration is used, this environment variable may not be set.
# Uncomment and modify it when you specified installation path of ProtoBuf.
# export CPLUS_INCLUDE_PATH=/home/HwHiAiUser/protobuf_install/include/:$CPLUS_INCLUDE_PATH

# ASCEND_TENSOR_COMPILER_INCLUDE: The path of the header file of the ATC package, where "/usr/local/Ascend/atc/include" is the
#                                 default installation path. If user defines the installation path, please modify it.
# Uncomment and modify it when you specified installation path of ATC.
# export ASCEND_TENSOR_COMPILER_INCLUDE=/usr/local/Ascend/atc/include

# The SYSTEM_INFO flag is the name of the form of the operator package generated by the compiler, which can be customized by the user, E.g:
#   a. If the operating system version is CentOS and the architecture is aarch64, it can be set to centos_aarch64, and the operator package name generated by compilation is custom_opp_centos_aarch64.run.
#   b. If the operating system version is CentOS and the architecture is x86_64, it can be set to centos_x86_64, and the name of the operator package generated by compilation is custom_opp_centos_x86_64.run.
#   c. If the SYSTEM_INFO environment variable is not set, the default value is used: ubuntu_x86_64, and the operator package name generated by compilation is custom_opp_ubuntu_x86_64.run.
# Uncomment and modify it when you need to specify os and architecture.
# export SYSTEM_INFO=ubuntu_x86_64

# If the installation path is not specified when installing ProtoBuf, the default configuration is used, and the following command is set to 'protoc' directly.
# If the installation path is specified when installing ProtoBuf, you need to use the absolute path of the protoc file.
PROTOC=protoc
# The install-path of the ATC package.
ATC_INSTALL_PATH=/usr/local/Ascend

###### The following logic can be used without modification ######

log() {
  cur_date=`date +"%Y-%m-%d %H:%M:%S"`
  echo "[$cur_date] "$1
}

project_path=$(cd "$(dirname "$0")"; pwd)
echo "--------------------------------"$project_path

if [ $1 == "clean" ] 2>/dev/null; then
  rm -rf $project_path/caffe.proto 2>/dev/null
  rm -rf $project_path/caffe.proto.origin 2>/dev/null
  rm -rf $project_path/build_out 2>/dev/null
  rm -rf $project_path/framework/caffe_plugin/proto/caffe 2>/dev/null
  log "[INFO] Clean successfully."
  exit 0
fi
chmod -R 755 $project_path/cmake/util/
mkdir -p $project_path/build_out

# check_is_digit() {
#   if [ $1 -gt 0 ] 2>/dev/null; then
#     return 0
#   else
#     log "[ERROR] Check digit failed. [$1] [$2]"
#     exit 1
#   fi
# }

# When the user-defined operator conflicts with the built-in operator,
# the user-defined operator is finally used.
# pre_proc() {
#   caffe_proto=$1
#   custom_proto=$2
#   log "[INFO] Pre proc for ${caffe_proto}."
#   declare_begin=$(grep -n -E "\s*message\s*+LayerParameter\s*+{|^\s*}" ${custom_proto} | grep -A 1 LayerParameter | head -1 | awk -F : '{print $1}')
#   declare_end=$(grep -n -E "\s*message\s*+LayerParameter\s*+{|^\s*}" ${custom_proto} | grep -A 1 LayerParameter | tail -1 | awk -F : '{print $1}')
#   check_is_digit $declare_begin "Invaliend begin of message LayerParameter."
#   check_is_digit $declare_end "Invaliend end of message LayerParameter."
#   declare_params=$(sed -n "$declare_begin, $declare_end p" ${custom_proto} | grep optional | awk -F ' ' '{print $2}')
#   log "[INFO] User-declared operator: $declare_params."

#   list=($(grep -E "\s*message.*{.*" ${custom_proto} | grep -v 'LayerParameter' | awk -F ' ' '{print $2}'))
#   log "[INFO] User-defined operator size: ${#list[@]}."
#   for list_element in ${list[@]}; do
#     element=$(echo $list_element | sed 's/{$//')
#     match_regex="\s*message\s*+"${element}"\s*+{"
#     match_count=$(grep -c -E $match_regex ${caffe_proto})
#     if [ "$match_count" == "0" ]; then
#       continue
#     fi
#     match_count=$(echo $declare_params | grep -c -w $element)
#     if [ "$match_count" == "1" ]; then
#       # Delete the declare in caffe_proto
#       declare_regex="\s*optional\s*+"${element}"\s.*"
#       declare_line=$(grep -n -E $declare_regex ${caffe_proto} | awk -F : '{print $1}')
#       check_is_digit $declare_line "Invaliend line number of declare."
#       sed -i ''$declare_line' d' ${caffe_proto}
#       if [ $? -ne 0 ]; then
#         log "[ERROR] Delete the declare of $element failed."
#         return 1
#       fi
#       log "[INFO] Delete the declare of $element."
#     fi
#     # Delete the definition in caffe_proto
#     define_regex="\s*message\s*+"${element}"\s*+{|^\s*}"
#     define_begin=$(grep -n -E $define_regex ${caffe_proto} | grep -A 1 message | head -1 | awk -F : '{print $1}')
#     define_end=$(grep -n -E $define_regex ${caffe_proto} | grep -A 1 message | tail -1 | awk -F : '{print $1}')
#     check_is_digit $declare_begin "Invaliend define_begin of definition."
#     check_is_digit $declare_end "Invaliend define_end of definition."
#     sed -i ''$define_begin','$define_end' d' ${caffe_proto}
#     if [ $? -ne 0 ]; then
#       log "[ERROR] Delete the definition of $element failed."
#       return 1
#     fi
#     log "[INFO] Delete the definition of $element."
#   done
#   log "[INFO] End of pre proc for ${caffe_proto}."
# }

# merge_proto() {
#   caffe_proto=$1
#   custom_proto=$2
#   pre_proc ${caffe_proto} ${custom_proto}
#   if [ $? -ne 0 ]; then
#     log "[ERROR] When the user-defined operator conflicts with the built-in operator, "
#         "the user-defined operator is finally used. Pre proc for this failed."
#     return 1
#   fi
#   begin=$(grep -n -E "^message\s+LayerParameter\s+{|^\s*}" ${custom_proto} | grep -A 1 LayerParameter | head -1 | awk -F : '{print $1}')
#   end=$(grep -n -E "^message\s+LayerParameter\s+{|^\s*}" ${custom_proto} | grep -A 1 LayerParameter | tail -1 | awk -F : '{print $1}')
#   check_is_digit $begin "Can not find message LayerParameter in custom proto."
#   check_is_digit $end "Can not find end of message LayerParameter in custom proto."
#   ((begin+=1))
#   ((end-=1))
#   check_is_digit $end "Invaliend end of message LayerParameter."
#   sed -n "$begin, $end p" ${custom_proto} > $project_path/build_out/insert.proto
#   if [ $? -ne 0 ]; then
#     log "[ERROR] Get LayerParameter from custom proto failed."
#     return 1
#   fi
#   ((end+=2))
#   insert_begin=$(grep -n -E "^message\s+LayerParameter\s+{|^\s*}" ${caffe_proto} | grep -A 1 LayerParameter | tail -1 | awk -F : '{print $1}')
#   check_is_digit $insert_begin "Failed to find insert position in caffe proto."
#   ((insert_begin-=1))
#   sed "$insert_begin r $project_path/build_out/insert.proto" ${caffe_proto} > $project_path/build_out/caffe.proto
#   if [ $? -ne 0 ]; then
#     log "[ERROR] Set LayerParameter to caffe proto failed."
#     return 1
#   fi
#   sed -n "$end, $ p" ${custom_proto} >> $project_path/build_out/caffe.proto
#   if [ $? -ne 0 ]; then
#     log "[ERROR] Set definition of Parameter to caffe proto failed."
#     return 1
#   fi
#   rm -rf $project_path/build_out/insert.proto
#   mv $project_path/build_out/caffe.proto ${caffe_proto}
# }

# STEP 1, Check plugin files and proto files
# Check for caffe_plugin files, If there is no plugin file, terminate subsequent operations.
cpp_files_num=$(ls $project_path/framework/caffe_plugin/*.cpp 2> /dev/null | wc -l)
if [ "$cpp_files_num" == "0" ]; then
  log "[INFO] No caffe plugin files, please check."
  exit 0
fi

# Check for caffe.proto, you can get it from ATC_INSTALL_PATH/atc/include/proto
# mkdir -p $project_path/framework/caffe_plugin/proto/caffe
# if [ -f "$project_path/caffe.proto.origin" ]; then
#   log "[INFO] Restore caffe.proto."
#   cp $project_path/caffe.proto.origin $project_path/caffe.proto
# fi
# if [ ! -f "$project_path/caffe.proto" ]; then
#   log "[INFO] No caffe.proto file in $project_path, copy it from $ATC_INSTALL_PATH/atc/include/proto."
#   cp $ATC_INSTALL_PATH/atc/include/proto/caffe.proto $project_path/caffe.proto
#   if [ $? -ne 0 ]; then
#     log "[ERROR] Copy caffe.proto file failed, you need to put origin caffe.proto here ($project_path) manually."
#     exit 1
#   fi
# fi
# if [ ! -f "$project_path/caffe.proto.origin" ]; then
#   log "[INFO] Backup caffe.proto."
#   cp $project_path/caffe.proto $project_path/caffe.proto.origin
# fi

# # Check for custom.proto
# if [ ! -f "$project_path/custom.proto" ]; then
#   log "[ERROR] No custom.proto file in $project_path, please check and add your op definition."
#   exit 1
# fi

# # STEP 2, Merge and compile proto
# log "[INFO] Merge caffe proto."
# merge_proto $project_path/caffe.proto $project_path/custom.proto
# if [ $? -ne 0 ]; then
#   log "[ERROR] Merge proto failed, please check your custom proto."
#   exit 1
# fi

# log "[INFO] Compile caffe proto."
# $PROTOC -I$project_path --cpp_out=$project_path/framework/caffe_plugin/proto/caffe/ $project_path/caffe.proto
# if [ $? -ne 0 ]; then
#   log "[ERROR] Protoc run failed, maybe need to compare the merged proto with origin or check your custom proto."
#   exit 1
# fi
