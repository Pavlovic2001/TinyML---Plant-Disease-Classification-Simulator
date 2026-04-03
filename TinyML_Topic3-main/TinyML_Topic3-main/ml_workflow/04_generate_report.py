# File: 04_generate_report.py




import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import os
import math



from config import (IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, TEST_DIR, 
                    MODEL_KERAS_PATH, HISTORY_PATH, REPORTS_DIR)

# IMG_WIDTH = 96
# IMG_HEIGHT = 96
# BATCH_SIZE = 32
# TEST_DIR = './test'
# MODEL_PATH = "saved_model/tomato_classifier.keras"
# HISTORY_PATH = "saved_model/training_history.npy"

# REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True) # Create the directory if it doesn't exist

print("--- Loading all necessary assets (model, data, history)... ---")

if not all(os.path.exists(p) for p in [MODEL_KERAS_PATH, HISTORY_PATH]):
    print(f"Error: Model or History file not found. Please run '02_train.py' first.")
    exit()

model = tf.keras.models.load_model(MODEL_KERAS_PATH)
history = np.load(HISTORY_PATH, allow_pickle=True).item()

def preprocess(image, label):
    image = tf.image.rgb_to_grayscale(image)
    image = image / 255.0
    return image, label

# === FIX: Load the dataset in batches, then concatenate them ===
# Load the dataset with a standard batch size first
test_ds_batched = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, seed=123, image_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE, # Use a valid batch size
    color_mode='rgb', shuffle=False
)
class_names = test_ds_batched.class_names
display_class_names = ['Healthy' if name == 'healthy' else 'Diseased' for name in class_names]

# Manually iterate through the dataset to collect all images and labels
print("--- Concatenating all test batches into a single array... ---")
all_images = []
all_labels = []
for images, labels in test_ds_batched:
    all_images.append(images.numpy())
    all_labels.append(labels.numpy())

# Concatenate the list of batches into single numpy arrays
test_images = np.concatenate(all_images)
y_true = np.concatenate(all_labels)

# Preprocess the concatenated images
test_images_processed, _ = preprocess(tf.convert_to_tensor(test_images), tf.convert_to_tensor(y_true))
# === END FIX ===

print("--- All assets loaded successfully. ---")


print("--- Generating predictions and metrics... ---")
y_pred_probs = model.predict(test_images_processed, verbose=0)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print("--- Creating the performance summary dashboard... ---")

fig, axes = plt.subplots(3, 2, figsize=(20, 24))
fig.suptitle('CNN Classifier Performance Summary', fontsize=28, fontweight='bold')

# ... (All the plotting code remains exactly the same as before) ...
ax1 = axes[0, 0]
epochs_ran = len(history['loss'])
ax1.plot(range(epochs_ran), history['accuracy'], color='blue', linestyle='-', label='Training Accuracy')
ax1.plot(range(epochs_ran), history['val_accuracy'], color='cyan', linestyle='--', label='Validation Accuracy')
ax1.plot(range(epochs_ran), history['loss'], color='red', linestyle='-', label='Training Loss')
ax1.plot(range(epochs_ran), history['val_loss'], color='orange', linestyle='--', label='Validation Loss')
ax1.set_title('Model Training History', fontsize=18)
ax1.legend()
ax1.grid(True, alpha=0.5)

ax2 = axes[0, 1]
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar_kws={'label': 'Count'})
ax2.set_title('Confusion Matrix', fontsize=18)
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')
ax2.set_xticklabels(display_class_names)
ax2.set_yticklabels(display_class_names)

ax3 = axes[1, 0]
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)
ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax3.set_title('ROC Curve', fontsize=18)
ax3.legend(loc="lower right")
ax3.grid(True, alpha=0.5)

ax4 = axes[1, 1]
precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
ax4.plot(recall, precision, color='purple', lw=2, label='Precision-Recall curve')
ax4.set_title('Precision-Recall Curve', fontsize=18)
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.legend(loc="lower left")
ax4.grid(True, alpha=0.5)

ax5 = axes[2, 0]
sns.kdeplot(y_pred_probs.flatten(), ax=ax5, fill=True, clip=(0, 1))
ax5.set_title('Prediction Confidence Distribution', fontsize=18)
ax5.set_xlabel('Predicted Probability (1 = Diseased)')
ax5.set_ylabel('Density')
ax5.grid(True, alpha=0.5)

ax6 = axes[2, 1]
report = classification_report(y_true, y_pred, target_names=display_class_names)
ax6.axis('off')
ax6.set_title('Performance Metrics', fontsize=18, y=0.8)
ax6.text(0.5, 0.5, report, ha='center', va='center', fontsize=16, fontfamily='monospace')

plt.tight_layout(rect=[0, 0, 1, 0.97])

# === NEW FEATURE: Save the dashboard to a file before showing it ===
dashboard_path = os.path.join(REPORTS_DIR, 'performance_summary.png')
print(f"--- Saving performance dashboard to {dashboard_path}... ---")
plt.savefig(dashboard_path)
plt.show()


# Visualize and Save Misclassified Images
print("\n--- Identifying and visualizing misclassified images... ---")
misclassified_indices = np.where(y_pred != y_true)[0]
num_to_show = min(len(misclassified_indices), 10)

if num_to_show > 0:
    misclassified_fig = plt.figure(figsize=(15, 5))
    misclassified_fig.suptitle(f'Showing {num_to_show} Misclassified Images', fontsize=16)
    for i, idx in enumerate(misclassified_indices[:num_to_show]):
        ax = plt.subplot(2, 5, i + 1)
        img = test_images[idx].astype("uint8") # Raw images are already in uint8
        ax.imshow(img)
        true_label = display_class_names[y_true[idx]]
        pred_label = display_class_names[y_pred[idx]]
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color='red')
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the misclassified images plot to a file
    misclassified_path = os.path.join(REPORTS_DIR, 'misclassified_examples.png')
    print(f"--- Saving misclassified examples plot to {misclassified_path}... ---")
    plt.savefig(misclassified_path)
    plt.show()
else:
    print("--- No misclassified images found in the test set. Excellent model! ---")

print("\n--- Report generation complete. ---")