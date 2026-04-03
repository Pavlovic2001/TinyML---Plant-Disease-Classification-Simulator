# 02_train.py

import sys
from pathlib import Path

# --- Add the project root directory to the Python path ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import os
import numpy as np # Import numpy to save the history

# --- Import all settings from the central config file ---
from config import (IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, TRAIN_DIR, VALIDATION_DIR, 
                    EPOCHS, MODEL_KERAS_PATH, HISTORY_PATH, DROPOUT_RATE, 
                    EARLY_STOPPING_PATIENCE)


# IMG_WIDTH = 96
# IMG_HEIGHT = 96
# BATCH_SIZE = 32
# TRAIN_DIR = './train'
# # TEST_DIR = './test' 
# VALIDATION_DIR = './validation' # Use the new validation set
# EPOCHS = 50 
# MODEL_SAVE_PATH = "saved_model/tomato_classifier.keras"
# HISTORY_SAVE_PATH = "saved_model/training_history.npy" 

# Data Loading and Preprocessing
print("--- Loading and preprocessing data... ---")

# Load training dataset from './train'
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, color_mode='rgb'
)

# # Load test/validation dataset from './test'
# test_ds = tf.keras.utils.image_dataset_from_directory(
#     TEST_DIR, seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE, color_mode='rgb'
# )

# Load validation dataset from './validation'
validation_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR, seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE, color_mode='rgb'
)

# Preprocessing function to convert to grayscale and normalize
def preprocess(image, label):
    image = tf.image.rgb_to_grayscale(image)
    image = image / 255.0
    return image, label

train_ds = train_ds.map(preprocess)
# test_ds = test_ds.map(preprocess)
validation_ds = validation_ds.map(preprocess)

# Optimize performance with caching and prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("--- Data preparation complete. ---")


print("\n--- Building the optimized classification model... ---")

inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))

# Convolutional Base
x = layers.Conv2D(16, 3, padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D()(x)

# Classifier Head
x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(DROPOUT_RATE)(x) 
outputs = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True
)

print("\n--- Starting model training... ---")
history = model.fit(
  train_ds,
#   validation_data=test_ds,
  validation_data=validation_ds, # Use the dedicated validation dataset
  epochs=EPOCHS,
  callbacks=[early_stopping] 
)
print("\n--- Model training complete. ---")

print(f"\n--- Saving the best model to {MODEL_KERAS_PATH}... ---")
os.makedirs(os.path.dirname(MODEL_KERAS_PATH), exist_ok=True)
model.save(MODEL_KERAS_PATH)
print("--- Model saved successfully. ---")
print(f"\n--- Saving training history to {HISTORY_PATH}... ---")
np.save(HISTORY_PATH, history.history)
print("--- History saved successfully. ---")

