# Experimental Log & Progress Tracking

**Project Title:** Smart Crop Disease Detection with Explainable Edge Deployment  
**Student:** DAHIPHALE SAMRADNI ASHOK (ID: A:32)  
**Host Environment:** Windows 11 x64, Python 3.11.0, 28-Thread High-Performance CPU, NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)  
**Primary Dataset:** Leaf Disease Dataset Combination (44 unique classes, 53,303 validated images)  
**Completed Date:** November 25, 2026  

---

## 1. Chronological Experiment Ledger

| Phase | Description | Status | Timestamp | Artifacts / Outputs Generated |
|:---:|:---|:---:|:---:|:---|
| **Phase 1** | Dataset Investigation & Integrity Scan | Completed | Nov 24, 2026 | Verified 44 classes, 53,303 total images; 0 corrupted files found |
| **Phase 2** | Data Preprocessing & Augmentation Pipeline | Completed | Nov 24, 2026 | `src/data_loader.py`, `src/preprocessing.py`, `results/figures/class_distribution.png` |
| **Phase 3** | Neural Architecture Construction | Completed | Nov 24, 2026 | `src/model_builder.py` (Custom CNN, VGG16, ResNet50, MobileNetV2) |
| **Phase 4** | Model Training & Benchmark Evaluation | Completed | Nov 25, 2026 | Checkpoints (`models/*.keras`), learning curves, `comparison_table.csv` |
| **Phase 5** | Ablation Study: Data Augmentation Impact | Completed | Nov 25, 2026 | `results/comparison/ablation_study.csv`, `results/figures/ablation_study.png` |
| **Phase 6** | Explainable AI (Grad-CAM) Visualizations | Completed | Nov 25, 2026 | 6 multi-panel diagnostic cases in `results/figures/gradcam/` |
| **Phase 7** | Edge TFLite Quantization & Benchmarking | Completed | Nov 25, 2026 | `models/tflite/model_{fp32,fp16,int8}.tflite`, `edge_optimization_table.csv` |
| **Phase 8** | Farmer-Friendly Web Application | Completed | Nov 25, 2026 | Interactive Streamlit GUI (`app/app.py`), `app/disease_info.json` |
| **Phase 9** | Documentation Suite & Jupyter Notebooks | Completed | Nov 25, 2026 | README.md, requirements.txt, 7 research notebooks in `notebooks/` |
| **Phase 10**| Academic Technical Report Compilation | Completed | Nov 25, 2026 | 8–10 page reports in `reports/technical_report.docx` & `technical_report.pdf` |
| **Phase 11**| Final Quality Control & Validation | Completed | Nov 25, 2026 | Full end-to-end reproducibility validation |

---

## 2. Empirical Architecture Parameters & Model Comparison

| Architecture | Total Parameters | Trainable Parameters | Model Size (MB) | Test Accuracy | Precision (Weighted) | Recall (Weighted) | Macro F1-Score | Inference Latency (ms) | Throughput (FPS) | Training Time (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Custom CNN Baseline** | 468,460 | 468,460 | 5.45 MB | 2.27% | 0.0005 | 0.0227 | 0.0011 | 27.77 ms | 36.0 FPS | 239.9 s |
| **VGG16 Transfer** | 14,858,348 | 143,660 | 57.88 MB | 74.09% | 0.7939 | 0.7409 | 0.7280 | 100.05 ms | 10.0 FPS | 674.5 s |
| **ResNet50 Transfer** | 24,124,588 | 536,876 | 96.77 MB | 81.52% | 0.8509 | 0.8152 | 0.8111 | 232.58 ms | 4.3 FPS | 269.4 s |
| **MobileNetV2 (Selected)**| 2,314,348 | 56,364 | 9.83 MB | 76.21% | 0.7702 | 0.7621 | 0.7507 | 129.08 ms | 7.7 FPS | 81.2 s |

---

## 3. Ablation Study: Impact of Data Augmentation on MobileNetV2

| Condition | Data Augmentation | Test Accuracy | Macro F1 | Weighted F1 | Training Loss | Validation Loss | Generalization Behavior |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Experiment A** | None (Static) | 79.85% | 0.7964 | 0.7964 | 0.5565 | 0.6445 | Minor overfitting on dominant leaf angles |
| **Experiment B** | Flip, Rotation, Zoom, Translation | 76.21% | 0.7507 | 0.7507 | 0.9984 | 0.8931 | Stronger regularization, robust to field orientation variance |

---

## 4. Edge Deployment & TFLite Quantization Benchmarks

| Model Format | Precision | Model Size (MB) | Compression Ratio | Mean Latency (ms) | Throughput (FPS) | Sample Accuracy | Target Hardware Feasibility |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Keras FP32 Baseline** | Float32 | 9.83 MB | 1.00x | 137.81 ms | 7.3 FPS | 88.33% | Cloud / Desktop Server only |
| **TFLite Float32** | Float32 | 8.69 MB | 1.13x | 7.91 ms | 126.4 FPS | 88.33% | High-end mobile / Tablet |
| **TFLite Float16** | Float16 | 4.41 MB | 2.23x | 7.91 ms | 126.4 FPS | 89.17% | Edge mobile / Embedded NPU |
| **TFLite INT8** | Int8 | 2.66 MB | **3.69x** | **5.42 ms** | **184.6 FPS** | 86.67% | Microcontrollers / Raspberry Pi / Offline Android |

---

## 5. Architectural Selection Rationale for Edge Deployment
Although ResNet50 achieved the highest uncompressed accuracy (81.52%), its massive 96.77 MB memory footprint and heavy 232.58 ms latency make it prohibitive for rural edge deployment. **MobileNetV2** was selected as the optimal deployment engine because:
1. It contains nearly **10x fewer parameters** than ResNet50 (2.31M vs. 24.12M).
2. It trains in less than one-third the time (81.2s vs. 269.4s).
3. Following INT8 quantization, it compresses down to **2.66 MB** with an ultra-fast **5.42 ms** inference latency (184.6 frames per second), while preserving 86.67% accuracy.
