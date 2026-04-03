# File: 05_convert_to_tflite.py


import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


import tensorflow as tf
import os
from config import MODEL_KERAS_PATH, TRAIN_DIR, TFLITE_MODEL_PATH

# --- Parameters ---
# MODEL_PATH = "saved_model/tomato_classifier.keras"
# TRAIN_DIR = './train' 
# TFLITE_MODEL_SAVE_PATH = "saved_model/tomato_classifier_quantized.tflite"



print("--- 1. Loading the trained Keras model... ---")
if not os.path.exists(MODEL_KERAS_PATH):
    print(f"Error: Keras model not found at {MODEL_KERAS_PATH}. Please run 02_train.py first.")
    exit()
model = tf.keras.models.load_model(MODEL_KERAS_PATH)

# Defining the Representative Dataset Generator
def representative_dataset_gen():
    print("--- Providing representative dataset for quantization... ---")
    def preprocess_for_rep_data(image, label):
        image = tf.image.rgb_to_grayscale(image)
        image = image / 255.0
        return image, label
        
    rep_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, seed=123, image_size=(96, 96),
        batch_size=1, color_mode='rgb'
    ).map(lambda x, y: preprocess_for_rep_data(x, y)[0])
    
    for image in rep_ds.take(100):
        yield [image]

# Setting up the TFLite Converter
print("--- 2. Setting up the TFLite converter with full integer quantization... ---")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

print("--- 3. Converting the model... ---")
tflite_model_quantized = converter.convert()

print(f"--- 4. Saving the quantized TFLite model to {TFLITE_MODEL_PATH}... ---")
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model_quantized)

print("\n--- Conversion successful! ---")
print(f"Original Keras model size: ~{os.path.getsize(MODEL_KERAS_PATH) / 1024:.0f} KB")
print(f"Quantized TFLite model size: ~{len(tflite_model_quantized) / 1024:.0f} KB")