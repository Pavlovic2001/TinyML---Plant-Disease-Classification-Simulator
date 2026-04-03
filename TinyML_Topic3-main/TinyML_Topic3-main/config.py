# File: config.py
# Purpose: Central configuration file for the entire project.
# This file provides a single source of truth for all paths and parameters.

from pathlib import Path

# ==============================================================================
#  1. CORE PATHS
#     We define the absolute path to the project root, and then build all
#     other paths from it. This makes the project robust to where it's run from.
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent

# --- Source and Processed Data ---
SOURCE_DATA_DIR = ROOT_DIR / "archive" / "PlantVillage"
TRAIN_DIR = ROOT_DIR / "train"
VALIDATION_DIR = ROOT_DIR / "validation"
TEST_DIR = ROOT_DIR / "test"

# --- Generated Assets ---
METADATA_DIR = ROOT_DIR / "metadata"
REPORTS_DIR = ROOT_DIR / "reports"
SAVED_MODEL_DIR = ROOT_DIR / "saved_model"

# ==============================================================================
#  2. FILE PATHS
# ==============================================================================
METADATA_CSV_PATH = METADATA_DIR / "test_metadata.csv"
MODEL_KERAS_PATH = SAVED_MODEL_DIR / "tomato_classifier.keras"
HISTORY_PATH = SAVED_MODEL_DIR / "training_history.npy"
TFLITE_MODEL_PATH = SAVED_MODEL_DIR / "tomato_classifier_quantized.tflite"

# ==============================================================================
#  3. MODEL & TRAINING PARAMETERS
# ==============================================================================
# --- Image Preprocessing ---
IMG_WIDTH = 96
IMG_HEIGHT = 96
BATCH_SIZE = 32

# --- Training ---
EPOCHS = 50 
EARLY_STOPPING_PATIENCE = 5
DROPOUT_RATE = 0.5

# --- Metadata Generation ---
LATITUDE_RANGE = (38.5, 39.5)   # Kansas, USA
LONGITUDE_RANGE = (-99.0, -100.0)




# 1. Serial port settings
SERIAL_PORT = '/dev/cu.usbmodem11301' 
BAUD_RATE = 115200

# 2. Communication protocol settings (must match the Arduino sketch)
CHUNK_SIZE = 32


