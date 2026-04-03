import sys
import serial
import time
from pathlib import Path
from PIL import Image
import numpy as np
import re
from sklearn.metrics import classification_report, confusion_matrix # Import new tools

# ==============================================================================
# --- Configuration (Keep unchanged) ---
# ==============================================================================
SERIAL_PORT = '/dev/cu.usbmodem11201'
BAUD_RATE = 115200
IMG_WIDTH = 64
IMG_HEIGHT = 64
FRAME_SIZE = IMG_WIDTH * IMG_HEIGHT
CHUNK_SIZE = 32

try:
    project_root = Path(__file__).resolve().parents[1]
except IndexError:
    project_root = Path.cwd()

sys.path.append(str(project_root))
TEST_DIR = project_root / "test"

# ==============================================================================
# --- Helper Functions (Keep unchanged) ---
# ==============================================================================

def preprocess_image(image_path: Path) -> bytes:
    """
    Opens an image, converts it to grayscale, resizes it, 
    and returns its byte representation.
    """
    img = Image.open(image_path)
    img_gray = img.convert('L')
    img_resized = img_gray.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
    return np.array(img_resized).tobytes()

def parse_prediction(lines: list) -> dict:
    """
    Parses the prediction output from the device to extract class labels and probabilities.
    """
    for line in lines:
        if line.startswith("Prediction ->"):
            pairs = re.findall(r"(\w+): ([\d\.]+)", line)
            return {label: float(value) for label, value in pairs}
    return {}

def run_single_inference(ser: serial.Serial, image_bytes: bytes) -> dict:
    """
    Sends an image to the device and retrieves the inference result.
    """
    ser.write(b'S')
    response = ser.readline().decode().strip()
    if response != "READY": 
        raise ConnectionError(f"Handshake failed: Expected 'READY', got '{response}'")

    num_chunks = FRAME_SIZE // CHUNK_SIZE
    for i in range(num_chunks):
        chunk = image_bytes[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        ser.write(chunk)
        ack = ser.readline().decode().strip()
        if ack != "ACK": 
            raise ConnectionError(f"Chunk {i+1} ACK failed: Expected 'ACK', got '{ack}'")
    
    result_lines = []
    while True:
        line = ser.readline().decode().strip()
        if not line: 
            raise TimeoutError("Timeout waiting for Arduino inference result.")
        result_lines.append(line)
        if "[END_OF_RESULT]" in line:
            break
    
    return parse_prediction(result_lines)

# ==============================================================================
# --- Main Program: Professional Evaluation ---
# ==============================================================================
if __name__ == "__main__":
    # 1. Collect specific test images as required
    print("Collecting test images...")
    healthy_paths = sorted(list((TEST_DIR / 'healthy').rglob('*.JPG'))) + sorted(list((TEST_DIR / 'healthy').rglob('*.jpg')))
    anomaly_paths = sorted(list((TEST_DIR / 'anomaly').rglob('*.JPG'))) + sorted(list((TEST_DIR / 'anomaly').rglob('*.jpg')))
    
    # Take only the first 200 anomaly images
    anomaly_paths_subset = anomaly_paths[:200]
    
    image_paths = healthy_paths + anomaly_paths_subset
    
    if not image_paths:
        print(f"Error: No images found in the directory {TEST_DIR}.")
        exit()
    
    print(f"Test set composition: {len(healthy_paths)} 'healthy' images + {len(anomaly_paths_subset)} 'anomaly' images = Total {len(image_paths)} images.")
    
    # 2. Initialize lists to store all results
    ground_truths = []
    predictions = []
    failed_inferences = 0

    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=10) as ser:
            print(f"Serial port {SERIAL_PORT} opened. Waiting for Arduino to get ready (2 seconds)...")
            time.sleep(2)
            ser.reset_input_buffer()

            for i, image_path in enumerate(image_paths):
                print(f"\n--- [ {i+1} / {len(image_paths)} ] ---")
                print(f"Testing: {image_path.name}")
                
                true_label = image_path.parent.name
                
                try:
                    image_bytes_to_send = preprocess_image(image_path)
                    prediction_dict = run_single_inference(ser, image_bytes_to_send)
                    
                    if not prediction_dict:
                        raise ValueError("Failed to parse prediction result.")

                    predicted_label = max(prediction_dict, key=prediction_dict.get)
                    
                    print(f"  True Label: {true_label}")
                    print(f"  Predicted Label: {predicted_label}")

                    # Store the true and predicted labels in the lists
                    ground_truths.append(true_label)
                    predictions.append(predicted_label)
                
                except Exception as e:
                    failed_inferences += 1
                    print(f"  => 💥 Inference failed: {e}")
                    ser.reset_input_buffer()

    except Exception as e:
        print(f"\nA fatal error occurred: {e}")
        exit()

    # 3. Calculate and print the final professional report
    print("\n\n======================================================")
    print("            Hardware-in-the-Loop Final Performance Report")
    print("======================================================")
    
    if not ground_truths:
        print("No inferences were successfully completed, cannot generate a report.")
        exit()

    # Get all unique labels and ensure consistent sorting
    class_labels = sorted(list(set(ground_truths)))

    # Print the confusion matrix
    cm = confusion_matrix(ground_truths, predictions, labels=class_labels)
    print("Confusion Matrix:")
    # Print the header
    header = " " * 12 + " ".join([f"{label:<10}" for label in class_labels])
    print(header)
    print(" " * 11 + "-" * (len(header) - 11))
    # Print each row
    for i, label in enumerate(class_labels):
        row_str = f"True {label:<6} |"
        for val in cm[i]:
            row_str += f" {val:<9}"
        print(row_str)
    print("\n")

    # Print the classification report
    print("Classification Report:")
    report = classification_report(ground_truths, predictions, labels=class_labels, digits=4)
    print(report)

    print(f"Total number of failed inferences: {failed_inferences}")
    print("======================================================")