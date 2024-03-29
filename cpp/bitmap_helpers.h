/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_EXAMPLES_OBJECT_DETECT_BITMAP_HELPERS_H_
#define TENSORFLOW_LITE_EXAMPLES_OBJECT_DETECT_BITMAP_HELPERS_H_

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_type.h"

using namespace std;
namespace tflite {
  namespace object_detect {
    
    vector<uint8_t> read_bmp(const string& input_bmp_name, int* width, int* height, int* channels);
    
    void resize(float* out, uint8_t* in, int image_height, int image_width,
		int image_channels, int wanted_height, int wanted_width,
		int wanted_channels, float input_mean, float input_std);
  }
}
#endif
