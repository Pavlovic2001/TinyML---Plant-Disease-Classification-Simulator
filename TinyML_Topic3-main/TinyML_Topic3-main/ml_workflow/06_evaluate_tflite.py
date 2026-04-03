# File: 06_evaluate_tflite.py


import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

from config import TEST_DIR, TFLITE_MODEL_PATH


print(f"--- 1. Loading TFLite model from {TFLITE_MODEL_PATH}... ---")
if not os.path.exists(TFLITE_MODEL_PATH):
    print(f"Error: TFLite model not found. Please run 05_convert_to_tflite.py first.")
    exit()

interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
print("--- TFLite model loaded and interpreter initialized. ---")


def preprocess(image, label):
    image = tf.image.rgb_to_grayscale(image)
    image = image / 255.0
    return image, label

test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, seed=123, image_size=(96, 96),
    batch_size=1, # Process one image at a time
    color_mode='rgb', shuffle=False
)
# extract the class names
class_names = test_ds_raw.class_names

# apply the preprocessing function to create the final dataset
test_ds = test_ds_raw.map(preprocess)
# Run inference on the entire test set ---
print("--- 4. Running inference on the test set... ---")
y_true = []
y_pred_probs = []

input_scale, input_zero_point = input_details['quantization']
output_scale, output_zero_point = output_details['quantization']
for image, label in test_ds:
    y_true.append(label.numpy()[0])
    
    image_quantized = tf.cast(image / input_scale + input_zero_point, dtype=input_details['dtype'])
    interpreter.set_tensor(input_details['index'], image_quantized)
    interpreter.invoke()
    
    output_quantized = interpreter.get_tensor(output_details['index'])[0][0]
    output_dequantized = (output_quantized.astype(np.float32) - output_zero_point) * output_scale
    y_pred_probs.append(output_dequantized)

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred = (y_pred_probs > 0.5).astype(int)

# --- 5. Display the performance report ---
print("\n--- Final Performance Report for Quantized TFLite Model ---")
print(classification_report(y_true, y_pred, target_names=class_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))