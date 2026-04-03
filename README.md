[README.md](https://github.com/user-attachments/files/26467348/README.md)
# TinyML Tomato Disease Classification Simulator

This project demonstrates a complete, end-to-end TinyML workflow for classifying healthy and diseased tomato plant leaves. It culminates in a dynamic, interactive web application that simulates a real-time drone survey, providing visual feedback on the model's performance.

The core of this project is a lightweight, fully integer-quantized TensorFlow Lite model (~590 KB) that achieves approximately 99% accuracy, making it ideal for deployment on resource-constrained edge devices.

## Technology Stack

-   **Backend & ML**: Python, TensorFlow, Keras, TensorFlow Lite, Scikit-learn, Pandas
-   **Frontend & Visualization**: Streamlit, Matplotlib

## Key Features

-   **Interactive Streamlit Dashboard**: A user-friendly web interface to configure and run the drone survey simulation.
-   **Dynamic Survey Simulation**: Generates a continuous and random flight path for each mission, simulating an intelligent exploration process.
-   **End-to-End ML Workflow**: A collection of organized scripts (`ml_workflow/`) that cover the entire process from data preparation and metadata generation to model training, evaluation, and conversion to TFLite.
-   **High-Performance Quantized Model**: The final model is a fully INT8 quantized `.tflite` file, demonstrating a >90% size reduction with negligible performance loss.

## File Structure
``` bash
├── app/ # Contains all code for the final Streamlit application
│ ├── dashboard.py
│ └── pipeline.py
│ └── test_arduino.py
├── arduino/ # Code for the microcontroller
│ └── Final_Classifier_HIL.ino # Arduino sketch for TFLite inference
│ └── model.h # delpoyed tflite model data
├── config.py # Central configuration file for all paths and parameters
├── ml_workflow/ # Scripts for the entire ML pipeline
│ ├── 00_generate_metadata.py
│ ├── 01_prepare_data.py
│ ├── 02_train.py
│ ├── 03_evaluate.py
│ ├── 04_generate_report.py
│ ├── 05_convert_to_tflite.py
│ └── 06_evaluate_tflite.py
│ └── 07_test_pipeline_accuracy.py
├── metadata/ # Stores the generated GPS metadata for the test set
├── reports/ # Stores generated performance charts
├── saved_model/ # Stores the trained Keras and TFLite models (.gitignore'd)
├── README.md
└── requirements.txt
```

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

-   Python 3.11 (Recommended)
-   An environment manager like `venv` or `conda`

### Setup & Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/yebeike/TinyML_Topic3
    cd TinyML_Topic3
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate
    
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```sh
    pip install -r
    ```

4.  **Download the Dataset:**
    -   Download the "New Plant Diseases Dataset" from Kaggle: [https://www.kaggle.com/datasets/emmarex/plantdisease/data](https://www.kaggle.com/datasets/emmarex/plantdisease/data)
    -   Unzip the downloaded `archive.zip` file.
    -   Place the unzipped contents so that the final folder structure is `./archive/PlantVillage/`.

## Usage / Workflow

Run the scripts **from the project root directory** in the specified order to reproduce the results.

1.  **Run the ML Workflow (Data Prep to Model Conversion):**
    These scripts will prepare the data, train the Keras model, evaluate it, and finally convert it to a quantized TFLite model.
    ```sh
    python ml_workflow/00_generate_metadata.py
    python ml_workflow/01_prepare_data.py
    python ml_workflow/02_train.py
    python ml_workflow/03_evaluate.py
    # Optional: python ml_workflow/04_generate_report.py

    # Because there is a problem with the 05 file (probably for the configuration environment), it crashed when running 05, so I uploaded the tflite file to Google Dirve, put it in the 'saved_models' folder, and then the rest of the py files can be run normally.
    # python ml_workflow/05_convert_to_tflite.py
    python ml_workflow/06_evaluate_tflite.py
    python ml_workflow/07_test_pipeline_accuracy.py
    ```

2.  **Launch the Simulator Application:**
    After the workflow is complete and the TFLite model is generated, launch the interactive dashboard.
    ```sh
    streamlit run app/dashboard.py
    ```
    Your web browser should open with the application running.
