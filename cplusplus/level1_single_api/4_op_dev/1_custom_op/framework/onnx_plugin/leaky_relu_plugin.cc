/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "graph/operator.h"
#include "register/register.h"
#include "json.hpp"

using namespace ge;
using json = nlohmann::json;

namespace domi {
Status ParseParamsLeakyRelu(const ge::Operator& op_src, ge::Operator& op_dest) {
  // trans op_src to op_dest
  // if op_src get required attr failed, need to return Failed
  // if op_src get optional attr failed, need to return Failed or set a default value
  float negative_slope = 0.01f;
  string attrs_string;
  if (ge::GRAPH_SUCCESS == op_src.GetAttr("attribute", attrs_string)) {
    json attrs = json::parse(attrs_string);
    for (json attr : attrs["attribute"]) {
      if (attr["name"] == "alpha" && attr["type"] == 1) {
        negative_slope = attr["f"];
      }
    }
  }

  op_dest.SetAttr("negative_slope", negative_slope);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("LeakyRelu")
    .FrameworkType(ONNX)
    .OriginOpType({"ai.onnx::8::LeakyRelu",
                   "ai.onnx::9::LeakyRelu",
                   "ai.onnx::10::LeakyRelu",
                   "ai.onnx::11::LeakyRelu",
                   "ai.onnx::12::LeakyRelu",
                   "ai.onnx::13::LeakyRelu"})
    .ParseParamsByOperatorFn(ParseParamsLeakyRelu)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
