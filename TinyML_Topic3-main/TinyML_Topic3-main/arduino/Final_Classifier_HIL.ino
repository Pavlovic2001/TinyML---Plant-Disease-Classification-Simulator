/*
 * Final Code - Direct TensorFlow Lite Inference
 *
 * This sketch ABANDONS the faulty Edge Impulse library and uses the official
 * Google TensorFlow Lite library to run the .tflite model directly.
 * This provides full transparency and control over the entire process.
 */

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>


#include "model.h"

// --- TFLite Global Variables ---
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;


constexpr int kTensorArenaSize = 60 * 1024; // leave some extra buffer
uint8_t tensor_arena[kTensorArenaSize];


const int FRAME_SIZE = 4096;
const int CHUNK_SIZE = 32;
uint8_t serial_buffer[FRAME_SIZE];

void setup() {
  Serial.begin(115200);
  delay(2000); 

  // --- Initialize TensorFlow Lite ---
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided has an unsupported schema version.");
    return;
  }

  // 2. Create an operator resolver (Op Resolver)
  static tflite::MicroMutableOpResolver<10> micro_op_resolver; 
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddAveragePool2D(); 
  micro_op_resolver.AddFullyConnected();

  // 3. Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // 4. Allocate memory for the model
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed.");
    return;
  }

  // 5. Get pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("TensorFlow Lite setup complete. Ready for inference.");
}

void loop() {
  if (Serial.available() > 0 && Serial.read() == 'S') {
    Serial.println("READY");

    int bytes_received = 0;
    while (bytes_received < FRAME_SIZE) {
      if (Serial.available() >= CHUNK_SIZE) {
        Serial.readBytes((char*)(serial_buffer + bytes_received), CHUNK_SIZE);
        bytes_received += CHUNK_SIZE;
        Serial.println("ACK");
      }
    }

    // --- [CORE] Interact directly with the TFLite model ---
    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;

    for (int i = 0; i < FRAME_SIZE; i++) {
      float pixel_float = (float)serial_buffer[i] / 255.0f;
      input->data.int8[i] = (int8_t)((pixel_float / input_scale) + input_zero_point);
    }

    // Run the model
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Invoke failed.");
      return;
    }

    int8_t anomaly_score_quantized = output->data.int8[0];
    int8_t healthy_score_quantized = output->data.int8[1];

    // Dequantize：Convert the int8 output back to a 0-1 floating-point probability
    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;
    float anomaly_prob = ((float)anomaly_score_quantized - output_zero_point) * output_scale;
    float healthy_prob = ((float)healthy_score_quantized - output_zero_point) * output_scale;

    // Send the results
    Serial.print("Prediction -> anomaly: "); Serial.print(anomaly_prob, 5);
    Serial.print(", healthy: "); Serial.println(healthy_prob, 5);
    Serial.println("[END_OF_RESULT]");
  }
}