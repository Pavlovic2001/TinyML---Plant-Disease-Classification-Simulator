# ==============================================================================
# File: app/dashboard.py (Version 2.0 - Cached)
# ==============================================================================

import streamlit as st
from pathlib import Path
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import sys

# --- [SETUP] Add project root and cached pipeline loading ---
# [MODIFICATION] This block is the ONLY change from the original v2.0
@st.cache_resource
def load_pipeline_cached():
    """Loads the AnomalyDetectionPipeline and caches it for the session."""
    try:
        project_root = Path(__file__).resolve().parents[1]
        sys.path.append(str(project_root))
        from app.pipeline import AnomalyDetectionPipeline
        # This print statement helps confirm the model is loaded only once.
        print("--- Initializing pipeline... ---")
        return AnomalyDetectionPipeline()
    except (ImportError, IndexError) as e:
        st.error(f"Could not load real pipeline due to: {e}. Using a dummy pipeline.")
        class DummyPipeline:
            def predict(self, path):
                is_anomaly = 'anomaly' in str(path)
                return {
                    'label': 'Diseased' if is_anomaly else 'Healthy',
                    'is_anomaly': is_anomaly,
                    'confidence': random.uniform(0.85, 1.0) if is_anomaly else random.uniform(0.9, 1.0),
                    'coords': {'latitude': 39.123, 'longitude': -98.456}
                }
        return DummyPipeline()

try:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root))
    from config import TEST_DIR
except (ImportError, IndexError):
    st.warning("Could not import TEST_DIR from config. Using a dummy path.")
    TEST_DIR = Path('.')

# Load the pipeline using the new cached function
pipeline = load_pipeline_cached()


# ==============================================================================
# 1. PAGE & STATE CONFIGURATION (Identical to v2.0)
# ==============================================================================
st.set_page_config(
    page_title="Drone Survey Simulator v2 (Cached)",
    page_icon="🛰️",
    layout="wide"
)

# --- Initialize Session State ---
if 'app_state' not in st.session_state:
    st.session_state.app_state = "INITIAL"
    st.session_state.log_messages = ["Welcome! Configure the survey and press 'Start'."]
    st.session_state.grid_size = 20
    st.session_state.flight_path = []
    st.session_state.image_map = {}
    st.session_state.ground_truth = {}
    st.session_state.predictions = {}
    st.session_state.current_step = 0
    st.session_state.mission_stats = {}
    st.session_state.anomalies_found = []


# ==============================================================================
# 2. CORE LOGIC & ALGORITHMS (Identical to v2.0)
# ==============================================================================

def generate_closed_loop_path(grid_size, start_cell, target_length):
    path = [start_cell]
    visited = {start_cell}
    r, c = start_cell
    possible_ends = []
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        end_cell = (r + dr, c + dc)
        if 0 <= end_cell[0] < grid_size and 0 <= end_cell[1] < grid_size:
            possible_ends.append(end_cell)
    
    if not possible_ends: return path 
    end_cell = random.choice(possible_ends)

    def _find_path_to_target(current_cell):
        if current_cell == end_cell and len(path) >= target_length: return True
        if len(path) >= target_length: return False
        r, c = current_cell
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]; random.shuffle(moves)
        for dr, dc in moves:
            next_cell = (r + dr, c + dc)
            if next_cell == end_cell and len(path) < target_length - 1: continue
            if 0 <= next_cell[0] < grid_size and 0 <= next_cell[1] < grid_size and next_cell not in visited:
                path.append(next_cell); visited.add(next_cell)
                if _find_path_to_target(next_cell): return True
                path.pop(); visited.remove(next_cell)
        return False

    if _find_path_to_target(start_cell):
        path.append(start_cell)
        return path
    else:
        st.session_state.log_messages.append("[WARN] Could not form perfect loop. Using best-effort path.")
        return path[:target_length]

def setup_new_survey(grid_size, num_points, seed):
    st.session_state.log_messages = ["[SYSTEM] Initializing new survey..."]
    st.session_state.flight_path = []
    st.session_state.image_map = {}
    st.session_state.ground_truth = {}
    st.session_state.predictions = {}
    st.session_state.current_step = 0
    st.session_state.mission_stats = {}
    st.session_state.anomalies_found = []
    st.session_state.grid_size = grid_size
    random.seed(seed)
    
    try:
        healthy_files = list((TEST_DIR / "healthy").glob("*.[jJ][pP][gG]"))
        anomaly_files = list((TEST_DIR / "anomaly").glob("*.[jJ][pP][gG]"))
        if not healthy_files and not anomaly_files: raise FileNotFoundError
    except FileNotFoundError:
        st.error(f"No images found. Using dummy data."); healthy_files = [f"dummy_h_{i}.jpg" for i in range(50)]; anomaly_files = [f"dummy_a_{i}.jpg" for i in range(50)]
    
    num_points = min(num_points, grid_size*grid_size, len(healthy_files)+len(anomaly_files))
    num_healthy = min(len(healthy_files), num_points//2); num_anomaly = min(len(anomaly_files), num_points-num_healthy)
    actual_num_points = num_healthy + num_anomaly
    selected_files = random.sample(healthy_files, k=num_healthy) + random.sample(anomaly_files, k=num_anomaly)
    random.shuffle(selected_files)

    st.session_state.log_messages.append(f"[SYSTEM] Generating path for {actual_num_points} points...")
    start_cell = (grid_size // 2, grid_size // 2)
    st.session_state.flight_path = generate_closed_loop_path(grid_size, start_cell, actual_num_points)
    
    if len(st.session_state.flight_path) > actual_num_points: st.session_state.flight_path = st.session_state.flight_path[:actual_num_points]
    if st.session_state.flight_path and st.session_state.flight_path[0] != st.session_state.flight_path[-1]: st.session_state.flight_path.append(st.session_state.flight_path[0])
    
    unique_cells = list(dict.fromkeys(st.session_state.flight_path))
    for i, cell in enumerate(unique_cells):
        if i < len(selected_files):
            file_path = selected_files[i]
            st.session_state.image_map[cell] = file_path
            st.session_state.ground_truth[cell] = "Healthy" if 'healthy' in str(file_path) else "Diseased"

    st.session_state.log_messages.append("[SYSTEM] Path generated. Drone ready.")
    st.session_state.app_state = "RUNNING"

def format_sector(cell):
    return f"{chr(ord('A') + cell[0])}-{cell[1] + 1}"

# ==============================================================================
# 3. UI RENDERING FUNCTIONS (Identical to v2.0)
# ==============================================================================

def draw_mission_configuration():
    st.header("1. Mission Configuration")
    grid_size = st.select_slider("Grid Size", options=[10, 15, 20], value=st.session_state.grid_size, disabled=(st.session_state.app_state == "RUNNING"))
    num_points = st.slider("Number of Points to Survey", min_value=10, max_value=grid_size*grid_size, value=40, disabled=(st.session_state.app_state == "RUNNING"))
    seed = st.number_input("Survey Plan Seed", min_value=0, value=42, disabled=(st.session_state.app_state == "RUNNING"))
    
    if st.button("🚀 Start New Survey", type="primary", disabled=(st.session_state.app_state == "RUNNING")):
        with st.spinner("Generating flight path..."):
            setup_new_survey(grid_size, num_points, seed)
        st.rerun()

def draw_live_feed_and_status():
    st.header("2. Live Drone Feed")
    feed_col1, feed_col2 = st.columns([1, 1])
    image_placeholder = feed_col1.empty()
    status_placeholder = feed_col2.empty()
    current_state = st.session_state.app_state
    
    if current_state == "INITIAL":
        image_placeholder.info("📸 Waiting for survey..."); status_placeholder.info("📊 Status will be displayed here.")
    elif current_state == "RUNNING":
        step = st.session_state.current_step
        if 0 <= step < len(st.session_state.flight_path):
            cell = st.session_state.flight_path[step]
            if cell in st.session_state.image_map:
                image_path = st.session_state.image_map[cell]
                prediction = st.session_state.predictions.get(cell)
                image_placeholder.image(str(image_path), use_container_width=True, caption=f"Analyzing Sector {format_sector(cell)}")
                if prediction:
                    conf = f"Confidence: {prediction['confidence']:.1%}"
                    if prediction['is_anomaly']: status_placeholder.metric("Prediction", "🚨 ANOMALY", conf)
                    else: status_placeholder.metric("Prediction", "✅ HEALTHY", conf)
                else: status_placeholder.info("Processing image...")
    elif current_state == "COMPLETE":
        stats = st.session_state.mission_stats
        image_placeholder.success(f"**Mission Complete!**"); image_placeholder.markdown(f"Final accuracy: **{stats.get('accuracy', 0):.2f}%**")
        if stats:
            labels = 'Correct', 'Incorrect'; sizes = [stats.get('correct', 0), stats.get('total', 0) - stats.get('correct', 0)]; colors = ['#4CAF50', '#F44336']
            fig, ax = plt.subplots(); ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90); ax.axis('equal')
            status_placeholder.pyplot(fig)

def draw_farm_health_map():
    st.header("3. Farm Health Map")
    map_placeholder = st.empty()
    grid_size = st.session_state.grid_size
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-0.5, grid_size - 0.5); ax.set_ylim(-0.5, grid_size - 0.5); ax.set_xticks(np.arange(0, grid_size, 2)); ax.set_yticks(np.arange(0, grid_size, 2)); ax.set_xticklabels([f"{i+1}" for i in np.arange(0, grid_size, 2)]); ax.set_yticklabels([chr(ord('A') + i) for i in np.arange(0, grid_size, 2)]); ax.invert_yaxis(); ax.set_aspect('equal')
    
    if st.session_state.app_state in ["RUNNING", "COMPLETE"]:
        for r in range(grid_size):
            for c in range(grid_size):
                truth = st.session_state.ground_truth.get((r, c))
                if truth: ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=('#d4edda' if truth == 'Healthy' else '#f8d7da'), edgecolor='grey', lw=0.5))
    
    for cell, pred in st.session_state.predictions.items():
        r, c = cell; symbol = '✓' if not pred['is_anomaly'] else '✗'; color = 'green' if not pred['is_anomaly'] else 'red'
        ax.text(c, r, symbol, fontsize=200 / grid_size, ha='center', va='center', color=color, weight='bold')

    path_so_far = st.session_state.flight_path[:st.session_state.current_step + 1]
    if path_so_far:
        path_coords = np.array(path_so_far)
        ax.plot(path_coords[:, 1], path_coords[:, 0], color='black', alpha=0.4, lw=1.5, zorder=3)

    if st.session_state.app_state == "RUNNING" and path_so_far:
        r, c = path_so_far[-1]
        ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='none', edgecolor='blue', linewidth=3, zorder=4))

    map_placeholder.pyplot(fig)

def draw_anomaly_report_and_logs():
    st.header("4. Reports & Logs")
    tab1, tab2 = st.tabs(["🚨 Anomaly Intelligence", "📜 Live Log"])
    with tab1:
        if st.session_state.app_state == "COMPLETE":
            if st.session_state.anomalies_found:
                st.subheader(f"Detected {len(st.session_state.anomalies_found)} Anomalies")
                for anomaly in st.session_state.anomalies_found:
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 2]); col1.image(str(anomaly['image_path']), use_container_width=True)
                        col2.metric("Sector", format_sector(anomaly['cell'])); col2.text(f"Confidence: {anomaly['confidence']:.1%}"); col2.text(f"GPS: {anomaly['coords']['latitude']:.4f}, {anomaly['coords']['longitude']:.4f}")
            else: st.success("✅ No anomalies were detected.")
        else: st.info("Anomaly report will be generated upon mission completion.")
    with tab2:
        log_container = st.container(height=400)
        for msg in st.session_state.log_messages:
            log_container.text(msg)

# ==============================================================================
# 4. MAIN APPLICATION FLOW (Identical to v2.0)
# ==============================================================================

st.title("TinyML Drone Survey Simulator")
st.markdown("---")
col_left, col_right = st.columns([1, 1.3], gap="large")

with col_left:
    draw_mission_configuration()
    st.markdown("---"); draw_live_feed_and_status()
with col_right:
    draw_farm_health_map()

st.markdown("---"); draw_anomaly_report_and_logs()

if st.session_state.app_state == "RUNNING":
    step = st.session_state.current_step; path = st.session_state.flight_path
    if step >= len(path) or not path:
        st.session_state.app_state = "COMPLETE"
        preds = st.session_state.predictions; ground_truth = st.session_state.ground_truth
        correct = sum(1 for c, p in preds.items() if ground_truth.get(c) == ("Healthy" if not p['is_anomaly'] else "Diseased"))
        total = len(preds)
        st.session_state.mission_stats = {'accuracy': (correct / total) * 100 if total > 0 else 100, 'correct': correct, 'total': total}
        st.session_state.log_messages.append("[SYSTEM] Mission Complete."); st.session_state.log_messages.append(f"[REPORT] Final Accuracy: {st.session_state.mission_stats['accuracy']:.2f}%")
        st.balloons(); st.rerun()
    else:
        current_cell = path[step]
        if current_cell not in st.session_state.predictions:
            image_path = st.session_state.image_map.get(current_cell)
            if image_path:
                st.session_state.log_messages.append(f"[DRONE] Analyzing sector {format_sector(current_cell)}...")
                result = pipeline.predict(image_path)
                st.session_state.predictions[current_cell] = result
                if result['is_anomaly']:
                    st.session_state.log_messages.append(f"   [!] ANOMALY DETECTED with {result['confidence']:.1%} confidence.")
                    st.session_state.anomalies_found.append({'cell': current_cell, 'image_path': image_path, **result})
        st.session_state.current_step += 1
        time.sleep(1.0) # Adjusted sleep time for a good balance
        st.rerun()