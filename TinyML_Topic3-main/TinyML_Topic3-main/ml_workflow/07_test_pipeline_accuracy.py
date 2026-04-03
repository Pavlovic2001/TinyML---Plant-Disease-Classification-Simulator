# File: ml_workflow/07_test_pipeline_accuracy.py

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from app.pipeline import AnomalyDetectionPipeline
from config import TEST_DIR

print("--- 1. Initializing the Anomaly Detection Pipeline... ---")
pipeline = AnomalyDetectionPipeline()

if pipeline.interpreter is None:
    print("Error: Pipeline failed to initialize. Exiting.")
    exit()

print("--- 2. Finding all images in the test directory... ---")
class_names = ['anomaly', 'healthy']

test_image_paths = []
image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
for ext in image_extensions:
    test_image_paths.extend(list(TEST_DIR.glob(f'*/**/{ext}')))

test_image_paths = sorted(list(set(test_image_paths)))


y_true = []
for path in test_image_paths:
    label_name = path.parent.name
    y_true.append(class_names.index(label_name))

# Run predictions on the entire test set using the pipeline
print(f"--- 3. Running inference on {len(test_image_paths)} test images... ---")
if len(test_image_paths) == 0:
    print("Error: No test images found. Please check the TEST_DIR path and image extensions.")
    exit()

y_pred = []
for image_path in tqdm(test_image_paths, desc="Processing images"):
    result = pipeline.predict(image_path)
    # is_anomaly=True is class 0, is_anomaly=False is class 1
    predicted_label = 0 if result['is_anomaly'] else 1
    y_pred.append(predicted_label)

# Convert to numpy arrays for sklearn
y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\n--- Final Performance Report for Production Pipeline ---")
print(classification_report(y_true, y_pred, target_names=class_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))