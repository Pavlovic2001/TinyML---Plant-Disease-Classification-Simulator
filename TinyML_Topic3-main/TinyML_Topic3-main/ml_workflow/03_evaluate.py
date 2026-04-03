# File: 03_evaluate.py (Upgraded for train/val/test split)



import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os


from config import IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, TEST_DIR, MODEL_KERAS_PATH


# IMG_WIDTH = 96
# IMG_HEIGHT = 96
# BATCH_SIZE = 32
# TEST_DIR = './test' # The final, untouched test set
# MODEL_PATH = "saved_model/tomato_classifier.keras"


print(f"--- Loading trained model from {MODEL_KERAS_PATH}... ---")
if not os.path.exists(MODEL_KERAS_PATH):
    print(f"Error: Model file not found at {MODEL_KERAS_PATH}")
    print("Please run '02_train.py' first to train and save the model.")
    exit()
model = tf.keras.models.load_model(MODEL_KERAS_PATH)
print("--- Model loaded successfully. ---")

def preprocess(image, label):
    image = tf.image.rgb_to_grayscale(image)
    image = image / 255.0
    return image, label

# Load the final test dataset from './test'
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, color_mode='rgb', shuffle=False
)
class_names = test_ds.class_names
print(f"Found classes for final evaluation: {class_names}")

test_ds = test_ds.map(preprocess).prefetch(buffer_size=tf.data.AUTOTUNE)

print("\n--- Performing final evaluation on the untouched test set... ---")

y_true = []
y_pred_probs = []
for images, labels in test_ds:
    y_true.extend(labels.numpy())
    y_pred_probs.extend(model.predict(images, verbose=0))

y_true = np.array(y_true)
y_pred = (np.array(y_pred_probs) > 0.5).astype(int).flatten()

print("\nClassification Report (on Final Test Set):")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\nConfusion Matrix (on Final Test Set):")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix on Final Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("\n--- Final evaluation complete. ---")