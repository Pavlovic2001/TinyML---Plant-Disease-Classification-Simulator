# File: ml_workflow/00_generate_metadata.py
import os
import csv
import random
import sys
from pathlib import Path

# --- Add project root to path ---
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from config import TEST_DIR, METADATA_DIR, METADATA_CSV_PATH, LATITUDE_RANGE, LONGITUDE_RANGE, ROOT_DIR

def generate_metadata():
    print("--- Starting GPS metadata generation for the test set... ---")
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    image_paths = list(TEST_DIR.glob("**/*.[jJ][pP][gG]"))
    if not image_paths:
        print(f"Error: No images found in '{TEST_DIR}'. Please run '01_prepare_data.py'.")
        return

    print(f"Found {len(image_paths)} images. Saving metadata to: {METADATA_CSV_PATH}")
    
    with open(METADATA_CSV_PATH, 'w', newline='') as csvfile:
        fieldnames = ['filepath', 'latitude', 'longitude']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for img_path in image_paths:
            lat = random.uniform(*LATITUDE_RANGE)
            lon = random.uniform(*LONGITUDE_RANGE)
            writer.writerow({
                'filepath': str(relative_path).replace('\\', '/'), # Use forward slashes for consistency
                'latitude': f"{lat:.6f}",
                'longitude': f"{lon:.6f}"
            })
            
    print(f"\n--- Successfully generated metadata for {len(image_paths)} images. ---")

if __name__ == "__main__":
    generate_metadata()