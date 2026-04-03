# File: 01_prepare_data.py



import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import os
import shutil
import random
from pathlib import Path

from config import (SOURCE_DATA_DIR, TRAIN_DIR, VALIDATION_DIR, TEST_DIR)

SOURCE_HEALTHY_FOLDER = "Tomato_healthy"
HEALTHY_CLASS_NAME = "healthy"
ANOMALY_CLASS_NAME = "anomaly"


print("--- Task: Preparing dataset with a Train/Validation/Test split ---")

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
# SOURCE_DATA_DIR = Path("./archive/PlantVillage")
# SOURCE_HEALTHY_FOLDER = "Tomato_healthy"
# BASE_DIR = Path(".") 
# TRAIN_DIR = BASE_DIR / "train"
# VALIDATION_DIR = BASE_DIR / "validation"
# TEST_DIR = BASE_DIR / "test"
# HEALTHY_CLASS_NAME = "healthy"
# ANOMALY_CLASS_NAME = "anomaly"


print("\n--- 1. Creating new train/validation/test directory structure ---")
for split_dir in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
    os.makedirs(split_dir / HEALTHY_CLASS_NAME, exist_ok=True)
    os.makedirs(split_dir / ANOMALY_CLASS_NAME, exist_ok=True)
print("Directory structure created successfully.")


def split_and_copy_files(source_input, category_name):
    """
    Finds all images from a source (folder or list of folders), splits them,
    and copies them to the corresponding target directories.
    """
    # === FIX START: The print statement is now more generic ===
    print(f"\n--- Processing all '{category_name}' data ---")
    # === FIX END ===
    
    files_to_process = []
    if isinstance(source_input, list):
        for folder in source_input:
            files_to_process.extend(list(folder.glob("*.[jJ][pP][gG]")))
    else:
        files_to_process = list(source_input.glob("*.[jJ][pP][gG]"))

    random.shuffle(files_to_process)
    total_files = len(files_to_process)
    
    train_end = int(total_files * TRAIN_RATIO)
    validation_end = train_end + int(total_files * VALIDATION_RATIO)
    
    train_files = files_to_process[:train_end]
    validation_files = files_to_process[train_end:validation_end]
    test_files = files_to_process[validation_end:]

    # Copy files
    target_train_dir = TRAIN_DIR / category_name
    print(f"Copying {len(train_files)} files to {target_train_dir}")
    for f in train_files:
        shutil.copy(f, target_train_dir)
        
    target_val_dir = VALIDATION_DIR / category_name
    print(f"Copying {len(validation_files)} files to {target_val_dir}")
    for f in validation_files:
        shutil.copy(f, target_val_dir)
        
    target_test_dir = TEST_DIR / category_name
    print(f"Copying {len(test_files)} files to {target_test_dir}")
    for f in test_files:
        shutil.copy(f, target_test_dir)


# Process Healthy and Anomaly Data
healthy_source_dir = SOURCE_DATA_DIR / SOURCE_HEALTHY_FOLDER
split_and_copy_files(healthy_source_dir, HEALTHY_CLASS_NAME)

anomaly_source_folders = []
for folder in SOURCE_DATA_DIR.iterdir():
    if folder.is_dir() and folder.name.startswith("Tomato_") and folder.name != SOURCE_HEALTHY_FOLDER:
        anomaly_source_folders.append(folder)
split_and_copy_files(anomaly_source_folders, ANOMALY_CLASS_NAME)


print("\n--- 5. Data splitting complete! Final file count: ---")
for split_name, split_dir in [("Training", TRAIN_DIR), ("Validation", VALIDATION_DIR), ("Test", TEST_DIR)]:
    healthy_count = len(list((split_dir / HEALTHY_CLASS_NAME).glob('*')))
    anomaly_count = len(list((split_dir / ANOMALY_CLASS_NAME).glob('*')))
    print(f"{split_name} set (healthy): {healthy_count} files")
    print(f"{split_name} set (anomaly): {anomaly_count} files")
    print("--------------------")

print("\nThe dataset is now ready with a train/validation/test split.")