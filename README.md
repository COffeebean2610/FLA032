# Smart Crop Disease Detection with Explainable Edge Deployment

**Academic Project ID:** A:32  
**Student Name:** DAHIPHALE SAMRADNI ASHOK  
**Department:** Computer Engineering / AI & Data Science  
**Academic Year:** 2026–2027  

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange.svg)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-red.svg)](https://streamlit.io)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

---

## 1. Project Overview & Problem Statement
Foliage diseases represent one of the primary drivers of yield loss and financial insecurity across global agricultural ecosystems. In developing agricultural economies like Maharashtra, India, farmers traditionally identify plant diseases through visual observation. This conventional approach is slow, highly subjective, prone to misdiagnosis, and inaccessible to smallholders in remote rural areas.

This project delivers an end-to-end, reproducible, research-grade machine learning system for automated plant disease diagnosis from leaf photographs. The system addresses two fundamental challenges in agricultural deep learning:
1. **Explainable AI (XAI):** Integrating Gradient-weighted Class Activation Mapping (Grad-CAM) to generate visual saliency heatmaps that highlight the exact foliar lesions and chlorotic margins driving the model's prediction.
2. **Edge Deployment Feasibility:** Optimizing the highest-performing architecture using TensorFlow Lite (Float32, Float16, and INT8 integer quantization) to enable real-time, low-latency inference on resource-constrained devices without requiring active internet connectivity.

---

## 2. Supported Crops and Regional Scope (Maharashtra Context)
The project utilizes the **Leaf Disease Dataset Combination** comprising **53,303 validated images across 44 discrete crop-disease and healthy classes**.

### Maharashtra-Relevant Crops Ground-Truth Audit:
- **Grape (*Vitis vinifera*):** 4 classes (Black Rot, Esca/Black Measles, Leaf Blight, Healthy) — *Major cash crop in Nashik/Sangli belts*.
- **Rice (*Oryza sativa*):** 4 classes (Brown Spot, Hispa, Leaf Blast, Healthy) — *Primary staple in Konkan and Vidarbha belts*.
- **Tomato (*Solanum lycopersicum*):** 10 classes (Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Mosaic Virus, Yellow Leaf Curl, Healthy).
- **Potato, Corn (Maize), and Bell Pepper:** Fully represented across multiple pathological conditions.
- **Honest Scoping Declaration:** Sugarcane (*Saccharum officinarum*) and Onion (*Allium cepa*) are absent from this public benchmark dataset. In accordance with strict academic integrity, this system does not claim or fabricate support for Sugarcane or Onion.

---

## 3. Technology Stack & Frameworks
- **Deep Learning Framework:** TensorFlow 2.21.0 & Keras 3.15.1
- **Computer Vision & Saliency:** OpenCV (`opencv-python`), Pillow (`PIL`), Grad-CAM
- **Data Science & Benchmarking:** NumPy, Pandas, Scikit-learn, SciPy
- **Visualization:** Matplotlib, Seaborn
- **Edge Inference:** TensorFlow Lite Runtime (`tf.lite`)
- **Web Interface:** Streamlit 1.67.0
- **Report Compilation:** Python-docx, ReportLab

---

## 4. Repository Structure
```text
Smart-Crop-Disease-Detection/
│
├── README.md                           # Master Project Documentation & Quickstart
├── requirements.txt                    # Pinned Python package dependencies
├── .gitignore                          # Git exclusions for large datasets & weights
│
├── notebooks/                          # 7 Sequential Research Jupyter Notebooks
│   ├── 01_dataset_exploration.ipynb    # Class audit, distribution, integrity scans
│   ├── 02_data_preprocessing.ipynb     # Pipeline, normalization, augmentation
│   ├── 03_custom_cnn.ipynb             # Baseline 4-stage CNN development
│   ├── 04_transfer_learning.ipynb      # VGG16, ResNet50, MobileNetV2 adaptation
│   ├── 05_model_comparison.ipynb       # Independent test set evaluation
│   ├── 06_ablation_study.ipynb         # Data augmentation impact analysis
│   └── 07_gradcam_explainability.ipynb # Grad-CAM saliency extraction
│
├── src/                                # Modular Production Python Source Code
│   ├── data_loader.py                  # High-throughput tf.data data loaders
│   ├── preprocessing.py                # Auditing, weights, distribution charts
│   ├── model_builder.py                # 4 deep neural network architectures
│   ├── train.py                        # Training engine with callbacks & scheduling
│   ├── evaluate.py                     # Multi-metric academic evaluation suite
│   ├── explainability.py               # Grad-CAM mathematical formulation
│   └── edge_optimization.py            # TFLite FP32, FP16, and INT8 quantizer
│
├── app/                                # Farmer-Friendly Web Application
│   ├── app.py                          # Streamlit interactive user interface
│   └── disease_info.json               # Symptoms & preventive guidelines (44 classes)
│
├── models/                             # Trained Checkpoints & Quantized Models
│   ├── best_model/                     # Optimal deployment Keras model
│   └── tflite/                         # Quantized edge deployment artifacts
│
├── results/                            # Experimental Artifacts & Figures
│   ├── figures/                        # Confusion matrices, curves, Grad-CAM plots
│   ├── metrics/                        # Classification reports & metrics JSON
│   └── comparison/                     # Consolidated benchmark tables
│
├── reports/                            # Submission-Ready Academic Documentation
│   ├── technical_report.docx           # 8–10 page formal IEEE-styled DOCX report
│   └── technical_report.pdf            # Compiled publication-grade PDF report
│
└── docs/                               # Formal Research Declarations
    ├── dataset_sources.md              # Dataset provenance and licensing
    ├── methodology.md                  # Complete mathematical formulation
    ├── ai_usage_declaration.md         # Academic integrity & AI tool disclosure
    └── experiment_log.md               # Chronological experimental ledger
```

---

## 5. Installation and Setup Instructions

### Prerequisites
- Python 3.11.x (64-bit)
- Windows 10/11, Linux, or macOS

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/<your-repo>/Smart-Crop-Disease-Detection.git
cd Smart-Crop-Disease-Detection

# Create virtual environment with Python 3.11
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### Step 2: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Dataset Download & Directory Setup
Download the dataset from Kaggle:
```python
import kagglehub
path = kagglehub.dataset_download("asheniranga/leaf-disease-dataset-combination")
print("Path to dataset:", path)
```
Extract or link the dataset into: `dataset/image data/` containing `train/`, `validation/`, and `test/` subdirectories.

---

## 6. Execution Workflow

### 1. Data Preparation & Class Distribution Analysis
```bash
python run_data_preparation.py
```
*Outputs: Generates `results/figures/class_distribution.png` and `results/metrics/class_names.json`.*

### 2. Model Training & Comparative Evaluation
```bash
python run_experiments.py
```
*Outputs: Trains all 4 architectures, evaluates on independent test data, runs the ablation study, and generates `results/comparison/comparison_table.csv` and `results/figures/model_comparison.png`.*

### 3. Explainable AI (Grad-CAM) Visualizations
```bash
python run_gradcam.py
```
*Outputs: Extracts class activation heatmaps and overlays for diagnostic cases in `results/figures/gradcam/`.*

### 4. Edge Quantization & Benchmarking
```bash
python run_edge_optimization.py
```
*Outputs: Exports FP32, FP16, and INT8 models in `models/tflite/` and benchmarks inference latency and compression in `results/comparison/edge_optimization_table.csv`.*

### 5. Launch the Farmer-Friendly Web Application
```bash
streamlit run app/app.py
```
*Access the local web dashboard at `http://localhost:8501`.*

### 6. Compile Technical Academic Report
```bash
python build_technical_report.py
```
*Generates publication-ready `reports/technical_report.docx` and `reports/technical_report.pdf`.*

---

## 7. Comparative Model Benchmark Summary

| Architecture | Test Accuracy | Precision (Weighted) | Recall (Weighted) | Macro F1-Score | Model Size (MB) | Inference Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Custom CNN Baseline** | 2.27% | 0.0005 | 0.0227 | 0.0011 | 5.45 MB | 27.77 ms |
| **VGG16 Transfer** | 74.09% | 0.7939 | 0.7409 | 0.7280 | 57.88 MB | 100.05 ms |
| **ResNet50 Transfer** | 81.52% | 0.8509 | 0.8152 | 0.8111 | 96.77 MB | 232.58 ms |
| **MobileNetV2 (Selected Edge Model)** | 76.21% | 0.7702 | 0.7621 | 0.7507 | 9.83 MB | 129.08 ms |

### Edge Optimization (TFLite Quantization) Summary:
- **Keras FP32 Baseline:** 9.83 MB | 137.81 ms Latency | 7.3 FPS | 88.33% Accuracy
- **TFLite Float32:** 8.69 MB (1.13x) | 7.91 ms Latency | 126.4 FPS | 88.33% Accuracy
- **TFLite Float16:** 4.41 MB (2.23x) | 7.91 ms Latency | 126.4 FPS | 89.17% Accuracy
- **TFLite INT8 Quantized:** **2.66 MB (3.69x)** | **5.42 ms Latency** | **184.6 FPS** | **86.67% Accuracy**

---

## 8. Responsible AI & Advisory Disclaimer
This machine learning system is engineered solely as an indicative decision-support prototype. Deep neural networks may exhibit diagnostic uncertainty under non-standardized field lighting, severe leaf occlusion, or camera blur. 

**Agricultural Warning:** The application does NOT prescribe chemical pesticides, fungicides, or specific dosages. Farmers and agricultural producers must consult a certified agronomist, plant pathologist, or local agricultural extension officer (e.g., Krishi Vigyan Kendra) before making chemical or treatment decisions.

---

## 9. Academic Citations
1. Hughes, D., & Salathé, M. (2015). An open access repository of images on plant health to enable mobile disease diagnostics. *arXiv:1511.08060*.
2. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.
3. Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, 145, 311–318.
4. Sandler, M., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *IEEE CVPR 2018*.
5. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *IEEE ICCV 2017*.
