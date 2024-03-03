#include <fcntl.h>
#include <getopt.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <typeinfo>
#include <math.h>
#include <algorithm>

#include "absl/memory/memory.h"
#include "object_detect/bitmap_helpers.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"

using namespace std;

namespace tflite {
  namespace object_detect {

    void softmax(float* input, int size) {

      int i;
      float m, sum, constant;

      m = -INFINITY;
      for (i = 0; i < size; ++i) {
        if (m < input[i]) {
          m = input[i];
        }
      }

      sum = 0.0;
      for (i = 0; i < size; ++i) {
        sum += exp(input[i] - m);
      }

      constant = m + log(sum);
      for (i = 0; i < size; ++i) {
        input[i] = exp(input[i] - constant);
      }
    }

    float sigmoid(float x){
      return 1.0 / (1.0 + exp(-x));
    }


    vector< vector<float> > processYoloLayer(TfLiteTensor* layer, int layerSize, int anchors[][2], 
                                             int numAnchors, int trainedImgSize, int numClasses, 
                                             float confThresh, vector< vector<float> > res) {
      const float* tensor_data = layer->data.f;
      vector< vector<float> > results = res;
      for (int p = 0; p < layer->dims->data[0]; p++) {
        for (int q = 0; q < layer->dims->data[1]; q++) {
          for (int r = 0; r < layer->dims->data[2]; r++){
            for (int s = 0; s < numAnchors; s++) {
              int offset = (numClasses + 5) * s;
              int base = p*(layer->dims->data[1]*layer->dims->data[2]*layer->dims->data[3]) + q*(layer->dims->data[2]*layer->dims->data[3]) + r*layer->dims->data[3];
              float confidence = sigmoid(tensor_data[base + offset + 4]);
              float* confidenceClasses = new float[numClasses];
              for (int c = 0; c < numClasses; c++){
                confidenceClasses[c] = tensor_data[base + offset + 5 + c];
              }
              softmax(confidenceClasses, numClasses);
              int objectId = max_element(confidenceClasses, confidenceClasses + numClasses) - confidenceClasses;
              float maxConf = confidenceClasses[objectId];
              confidence = confidence * maxConf;
              if (confidence > confThresh) {
                float x = (r + sigmoid(tensor_data[base + offset])) / layerSize;
                float y = (q + sigmoid(tensor_data[base + offset + 1])) / layerSize;
                float w = exp(tensor_data[base + offset + 2]) * anchors[s][0] / trainedImgSize;
                float h = exp(tensor_data[base + offset + 3]) * anchors[2][1] / trainedImgSize;
                float left = x - w / 2;
                float top = y - h / 2;
                float right = x + w / 2;
                float bottom = y + h / 2;
                if (results[objectId][4] < confidence) {
                  results[objectId][0] = x - w / 2;
                  results[objectId][1] = y - h / 2;
                  results[objectId][2] = x + w / 2;
                  results[objectId][3] = y + h / 2;
                  results[objectId][4] = confidence;
                }
                
              }
              delete [] confidenceClasses;
            }
          }
        }
      }
      return results;
    }

    
    void RunInference(string model_name, string image_name, float input_mean, float input_std) {
      // Setup the interpreter
      unique_ptr<tflite::FlatBufferModel> model;
      unique_ptr<tflite::Interpreter> interpreter;

      model = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
      model->error_reporter();

      tflite::ops::builtin::BuiltinOpResolver resolver;

      tflite::InterpreterBuilder(*model, resolver)(&interpreter);
      
      if (!interpreter) {
	      cout << "Failed to construct interpreter";
	      exit(-1);
      }
      
      interpreter->SetNumThreads(4);

      // Setup the inputs
      int image_width;
      int image_height;
      
      int image_channels;
      
      vector<uint8_t> in = read_bmp(image_name, &image_width,
				    &image_height, &image_channels);

      int input = interpreter->inputs()[0];
      
      if (interpreter->AllocateTensors() != kTfLiteOk) {
	      cout << "Failed to allocate tensors!";
	      exit(-1);
      }
      
      TfLiteIntArray* dims = interpreter->tensor(input)->dims;
      int wanted_height = dims->data[1];
      int wanted_width = dims->data[2];
      int wanted_channels = dims->data[3];
      
      resize(interpreter->typed_tensor<float>(input), in.data(),
	     image_height, image_width, image_channels, wanted_height,
	     wanted_width, wanted_channels, input_mean, input_std);

      // Invoke the interpreter
      if (interpreter->Invoke() != kTfLiteOk) {
	      cout << "Failed to invoke tflite!";
	      exit(-1);
      }

      // Get the outputs
      TfLiteTensor* layer_one = interpreter->tensor(interpreter->outputs()[0]);
      TfLiteTensor* layer_two = interpreter->tensor(interpreter->outputs()[1]);

      // layer_one and layer_two contain the two outputs from the sanitizer.
      // layer_one contains a float buffer of size 4563 (1 x 13 x 13 x 27).
      // layer_two contains a float buffer of size 18252 (1 x 26 x 26 x 27).

      int layerOneAnchors[][2] =  {{81, 82}, {135, 169}, {344, 319}};
      int layerTwoAnchors[][2] =  {{10, 14}, {23, 27}, {37, 58}};
      
      // since the model dectects 4 classes
      // 0: Background, 1: Card, 2: QR, 3: Face
      const int numLabels = 4;
      // each object consits of 4 cordinates and a confidence score
      // the class id is determined by the index in the vector datastructure.
      const int numVals = 5;
      
      const int trainedImgSize = 416;
      const float confidenceThreshold = 0.2;
      const int numAnchors = 3;
      const int layerOneSize = 13;
      const int layerTwoSize = 26;

      vector< vector<float> > results(numLabels);
      for (int i = 0; i < numLabels; i++) {
        results[i] = vector<float>(numVals);
        for (int j = 0; j < numVals; j++) {
          results[i][j] = -INFINITY;
        }
      }

      results = processYoloLayer(layer_one, layerOneSize, layerOneAnchors, numAnchors, trainedImgSize, numLabels, confidenceThreshold, results);
      results = processYoloLayer(layer_two, layerTwoSize, layerTwoAnchors, numAnchors, trainedImgSize, numLabels, confidenceThreshold, results);

      // Get the results for card class i.e left, top, right, bottom and confidence.

      cout << results[1][0]*image_width << endl;
      cout << results[1][1]*image_height << endl;
      cout << results[1][2]*image_width << endl;
      cout << results[1][3]*image_height << endl;
      cout << results[1][4] << endl;

      cout << "Done" << endl;
    }
  }
}

int main() {
  string model_name = "";
  string image_name = "";
  float input_mean = 0.0f, input_std = 255.0;
  tflite::object_detect::RunInference(model_name, image_name, input_mean, input_std);
  return 0;
}
