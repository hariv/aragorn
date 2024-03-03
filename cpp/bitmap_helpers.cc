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

#include "object_detect/bitmap_helpers.h"

#include <unistd.h>  // NOLINT(build/include_order)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"

using namespace std;
namespace tflite {
  namespace object_detect {
    
    void resize(float* out, uint8_t* in, int image_height, int image_width,
		int image_channels, int wanted_height, int wanted_width,
		int wanted_channels, float input_mean, float input_std) {
      int number_of_pixels = image_height * image_width * image_channels;
      unique_ptr<Interpreter> interpreter(new Interpreter);
      
      int base_index = 0;
      
      // two inputs: input and new_sizes
      interpreter->AddTensors(2, &base_index);
      // one output
      interpreter->AddTensors(1, &base_index);
      // set input and output tensors
      interpreter->SetInputs({0, 1});
      interpreter->SetOutputs({2});
      
      // set parameters of tensors
      TfLiteQuantizationParams quant;
      interpreter->SetTensorParametersReadWrite(
						0, kTfLiteFloat32, "input",
						{1, image_height, image_width, image_channels}, quant);
      interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
						quant);
      interpreter->SetTensorParametersReadWrite(
						2, kTfLiteFloat32, "output",
						{1, wanted_height, wanted_width, wanted_channels}, quant);
      
      ops::builtin::BuiltinOpResolver resolver;
      const TfLiteRegistration* resize_op =
	resolver.FindOp(BuiltinOperator_RESIZE_BILINEAR, 1);
      auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(
								   malloc(sizeof(TfLiteResizeBilinearParams)));
      params->align_corners = false;
      params->half_pixel_centers = false;
      interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,
					 nullptr);
      
      interpreter->AllocateTensors();
      
      // fill input image
      // in[] are integers, cannot do memcpy() directly
      auto input = interpreter->typed_tensor<float>(0);
      for (int i = 0; i < number_of_pixels; i++) {
	input[i] = in[i];
      }
      
      // fill new_sizes
      interpreter->typed_tensor<int>(1)[0] = wanted_height;
      interpreter->typed_tensor<int>(1)[1] = wanted_width;
      
      interpreter->Invoke();
      
      auto output = interpreter->typed_tensor<float>(2);
      auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;
      
      for (int i = 0; i < output_number_of_pixels; i++) {
	out[i] = (output[i] - input_mean) / input_std;
      }
    }
    
    vector<uint8_t> decode_bmp(const uint8_t* input, int row_size, int width,
				    int height, int channels, bool top_down) {
      vector<uint8_t> output(height * width * channels);
      
      for (int i = 0; i < height; i++) {
	int src_pos;
	int dst_pos;
	
	for (int j = 0; j < width; j++) {
	  if (!top_down) {
	    src_pos = ((height - 1 - i) * row_size) + j * channels;
	  } else {
	    src_pos = i * row_size + j * channels;
	  }
	  
	  dst_pos = (i * width + j) * channels;
	  
	  switch (channels) {
	  case 1:
	    output[dst_pos] = input[src_pos];
	    break;
	  case 3:
	    // BGR -> RGB
	    output[dst_pos] = input[src_pos + 2];
	    output[dst_pos + 1] = input[src_pos + 1];
	    output[dst_pos + 2] = input[src_pos];
	    break;
	  case 4:
	    // BGRA -> RGBA
	    output[dst_pos] = input[src_pos + 2];
	    output[dst_pos + 1] = input[src_pos + 1];
	    output[dst_pos + 2] = input[src_pos];
	    output[dst_pos + 3] = input[src_pos + 3];
	    break;
	  default:
	    cout << "Unexpected number of channels: " << channels;
	    break;
	  }
	}
      }
      return output;
    }
    
    vector<uint8_t> read_bmp(const string& input_bmp_name, int* width, int* height, int* channels) {
      int begin, end;
      
      ifstream file(input_bmp_name, ios::in | ios::binary);
      if (!file) {
	cout << "input file " << input_bmp_name << " not found";
	exit(-1);
      }
      
      begin = file.tellg();
      file.seekg(0, ios::end);
      end = file.tellg();
      size_t len = end - begin;
      
      vector<uint8_t> img_bytes(len);
      file.seekg(0, ios::beg);
      file.read(reinterpret_cast<char*>(img_bytes.data()), len);
      const int32_t header_size =
	*(reinterpret_cast<const int32_t*>(img_bytes.data() + 10));
      *width = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 18));
      *height = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 22));
      const int32_t bpp =
	*(reinterpret_cast<const int32_t*>(img_bytes.data() + 28));
      *channels = bpp / 8;
      
      // there may be padding bytes when the width is not a multiple of 4 bytes
      // 8 * channels == bits per pixel
      const int row_size = (8 * *channels * *width + 31) / 32 * 4;
      
      // if height is negative, data layout is top down
      // otherwise, it's bottom up
      bool top_down = (*height < 0);
      
      // Decode image, allocating tensor once the image size is known
      const uint8_t* bmp_pixels = &img_bytes[header_size];
      return decode_bmp(bmp_pixels, row_size, *width, abs(*height), *channels,
			top_down);
    }
    
  }
}
