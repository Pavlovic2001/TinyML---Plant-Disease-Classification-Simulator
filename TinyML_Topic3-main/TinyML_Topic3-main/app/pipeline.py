# File: app/pipeline.py (Final Corrected Version - With dtype fix and tf.cast fix)

import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# --- Add project root to path ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from config import (TFLITE_MODEL_PATH, METADATA_CSV_PATH, IMG_WIDTH, IMG_HEIGHT, 
                    ROOT_DIR, TEST_DIR)

class AnomalyDetectionPipeline:
    def __init__(self):
        print("--- Initializing pipeline... ---")
        try:
            self.metadata = pd.read_csv(METADATA_CSV_PATH).set_index('filepath')
            print("--- Metadata loaded successfully. ---")
        except FileNotFoundError:
            self.metadata = None
            print(f"Warning: Metadata file not found at {METADATA_CSV_PATH}. GPS coordinates will be unavailable.")

        try:
            self.interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]
            print("--- TFLite model loaded and interpreter initialized. ---")
        except Exception as e:
            self.interpreter = None
            print(f"Error loading TFLite model: {e}")

    def _preprocess_image(self, image_path):
        """
        Preprocesses an image file using TensorFlow operations to ensure 1:1 consistency
        with the training and evaluation pipeline.
        """
        img_raw = tf.io.read_file(str(image_path))
        img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
        # ---------------Cast to float32 *before* resizing----------------
        img = tf.cast(img, tf.float32) 
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.image.rgb_to_grayscale(img)
        img = img / 255.0
        return tf.expand_dims(img, axis=0)
    
    def predict(self, image_path):
        if self.interpreter is None:
            return {"error": "Pipeline not initialized correctly."}

        image_path = Path(image_path)
        image_tensor = self._preprocess_image(image_path)

        input_scale, input_zero_point = self.input_details['quantization']
        
        # === THE CRITICAL FIX for the AttributeError ===
        # Use tf.cast() for TensorFlow tensors, not .astype()
        quantization_result = (image_tensor / input_scale) + input_zero_point
        image_quantized = tf.cast(quantization_result, dtype=self.input_details['dtype'])
        
        self.interpreter.set_tensor(self.input_details['index'], image_quantized)
        self.interpreter.invoke()
        
        output_quantized = self.interpreter.get_tensor(self.output_details['index'])[0][0]
        output_scale, output_zero_point = self.output_details['quantization']
        
        prob_healthy = (output_quantized.astype(np.float32) - output_zero_point) * output_scale
        
        if prob_healthy > 0.5:
            is_anomaly = False
            label = "Healthy"
            confidence = prob_healthy
        else:
            is_anomaly = True
            label = "Diseased"
            confidence = 1 - prob_healthy

        coords = pd.Series({'latitude': 0.0, 'longitude': 0.0})
        if self.metadata is not None:
            try:
                relative_path_str = str(image_path.relative_to(ROOT_DIR)).replace('\\', '/')
                coords = self.metadata.loc[relative_path_str]
            except (KeyError, ValueError):
                pass
        
        return {
            "filepath": str(image_path),
            "is_anomaly": is_anomaly,
            "label": label,
            "confidence": float(confidence),
            "coords": {
                "latitude": coords['latitude'],
                "longitude": coords['longitude']
            }
        }

# Example usage for quick testing
if __name__ == '__main__':
    example_image_path = list(TEST_DIR.glob("anomaly/*"))[0]
    
    pipeline = AnomalyDetectionPipeline()
    if pipeline.interpreter:
        result = pipeline.predict(example_image_path)
        import json
        print("\n--- Pipeline Test Result ---")
        print(json.dumps(result, indent=2))