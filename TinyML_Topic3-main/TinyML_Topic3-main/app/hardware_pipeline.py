# File: app/hardware_pipeline.py (V23 - Stateless and Streamlit-Safe)

import serial
import time
import re
from pathlib import Path
import sys
import pandas as pd

# 确保能从 config.py 导入配置
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from config import (SERIAL_PORT, BAUD_RATE, METADATA_CSV_PATH, ROOT_DIR,
                    IMG_WIDTH, IMG_HEIGHT, CHUNK_SIZE)
from PIL import Image
import numpy as np

class HardwarePipeline:
    def __init__(self):
        """
        初始化 pipeline。注意：这里不再打开串口连接。
        """
        print("--- Initializing Stateless Hardware Pipeline instance... ---")
        try:
            self.metadata = pd.read_csv(METADATA_CSV_PATH).set_index('filepath')
        except FileNotFoundError:
            self.metadata = None

    def _preprocess_image(self, image_path: Path) -> bytes:
        img = Image.open(image_path)
        img_gray = img.convert('L')
        img_resized = img_gray.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        return np.array(img_resized).tobytes()

    def _parse_prediction(self, lines: list) -> dict:
        for line in lines:
            if line.startswith("Prediction ->"):
                pairs = re.findall(r"(\w+): ([\d\.]+)", line)
                return {label: float(value) for label, value in pairs}
        return {}

    def predict(self, image_path):
        """
        为单次预测处理完整的串口连接生命周期。
        这使得它对于 Streamlit 的 rerun 模型是安全的。
        """
        image_path = Path(image_path)
        
        # 【核心修正】'with' 语句确保每次预测都使用一个全新的连接，并在结束时自动关闭
        try:
            with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=10) as ser:
                # 这里的逻辑和我们成功的批量测试脚本完全一致
                time.sleep(2) # 等待 Arduino 重启并准备好
                ser.reset_input_buffer()

                image_bytes = self._preprocess_image(image_path)
                
                # 执行我们成功的 S/READY/ACK 协议
                ser.write(b'S')
                response = ser.readline().decode().strip()
                if response != "READY":
                    raise ConnectionError(f"Handshake failed: Expected 'READY', got '{response}'")

                frame_size = IMG_WIDTH * IMG_HEIGHT
                num_chunks = frame_size // CHUNK_SIZE
                for i in range(num_chunks):
                    chunk = image_bytes[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
                    ser.write(chunk)
                    ack = ser.readline().decode().strip()
                    if ack != "ACK":
                        raise ConnectionError(f"Chunk ACK failed at chunk {i+1}")
                
                # 接收结果
                result_lines = []
                while True:
                    line = ser.readline().decode().strip()
                    if not line:
                        raise TimeoutError("Timeout waiting for Arduino result.")
                    result_lines.append(line)
                    if "[END_OF_RESULT]" in line:
                        break
                
                prediction = self._parse_prediction(result_lines)
                if not prediction:
                    raise ValueError("Could not parse prediction from Arduino.")
                    
                prob_healthy = prediction.get('healthy', 0.0)
                is_anomaly = prob_healthy <= 0.5
                label = "Diseased" if is_anomaly else "Healthy"
                confidence = 1 - prob_healthy if is_anomaly else prob_healthy

        except Exception as e:
            print(f"ERROR during HIL prediction for {image_path.name}: {e}")
            return { "error": str(e), "label": "Error", "is_anomaly": True, "confidence": 1.0, "coords": {"latitude": 0, "longitude": 0} }

        # 获取 GPS 坐标 (这部分逻辑保持不变)
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