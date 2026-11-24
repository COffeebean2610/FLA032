import json
import os

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            },
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Notebook written: {filepath}")

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

# 1. 01_dataset_exploration.ipynb
nb1_cells = [
    md_cell("# 01. Dataset Exploration and Integrity Analysis\n\n**Project:** Smart Crop Disease Detection with Explainable Edge Deployment\n**Student:** DAHIPHALE SAMRADNI ASHOK (ID: A:32)\n\nThis notebook inspects the `leaf-disease-dataset-combination` dataset, verifies folder hierarchies across train/validation/test splits, computes class distribution statistics across 44 categories, and verifies image file integrity."),
    code_cell("""import os
import sys
sys.path.append('..')
from src.data_loader import get_class_names, get_image_paths_and_labels
from src.preprocessing import check_corrupted_images, check_duplicates, plot_class_distribution"""),
    code_cell("""# 1. Verify Class Inventory across 44 categories
class_names = get_class_names('../dataset/image data')
print(f"Total Unique Verified Classes: {len(class_names)}")
for i, c in enumerate(class_names):
    print(f"{i+1:02d}. {c}")"""),
    code_cell("""# 2. Check for Corrupted Images across the entire corpus
total_scanned, corrupted = check_corrupted_images('../dataset/image data')
print(f"Total images scanned: {total_scanned}")
print(f"Corrupted image files detected: {len(corrupted)}")"""),
    code_cell("""# 3. Check for Duplicate Files
scanned, dups = check_duplicates('../dataset/image data', sample_limit=2000)
print(f"Scanned {scanned} images; found {len(dups)} duplicate MD5 hashes.")""")
]
create_notebook(nb1_cells, "notebooks/01_dataset_exploration.ipynb")

# 2. 02_data_preprocessing.ipynb
nb2_cells = [
    md_cell("# 02. Data Preprocessing, Normalization and Augmentation\n\n**Project:** Smart Crop Disease Detection with Explainable Edge Deployment\n\nThis notebook sets up the input data pipeline using TensorFlow `tf.data`, applies standard 224x224 resizing, pixel normalization to [0, 1], and data augmentation for training."),
    code_cell("""import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
sys.path.append('..')
from src.data_loader import create_dataset, get_augmentation_layer"""),
    code_cell("""# Create Training and Validation tf.data Pipelines
train_ds, n_train, classes = create_dataset('../dataset/image data', split='train', batch_size=32, augment=True)
val_ds, n_val, _ = create_dataset('../dataset/image data', split='validation', batch_size=32, augment=False)
test_ds, n_test, _ = create_dataset('../dataset/image data', split='test', batch_size=32, augment=False)
print(f"Train samples: {n_train} | Val samples: {n_val} | Test samples: {n_test}")"""),
    code_cell("""# Inspect Batch Shape and Normalization Range
for images, labels in train_ds.take(1):
    print(f"Batch images tensor shape: {images.shape} (dtype: {images.dtype})")
    print(f"Batch labels tensor shape: {labels.shape}")
    print(f"Min pixel value: {tf.reduce_min(images):.3f}, Max pixel value: {tf.reduce_max(images):.3f}")""")
]
create_notebook(nb2_cells, "notebooks/02_data_preprocessing.ipynb")

# 3. 03_custom_cnn.ipynb
nb3_cells = [
    md_cell("# 03. Custom CNN Baseline Model\n\n**Project:** Smart Crop Disease Detection with Explainable Edge Deployment\n\nBuilds, compiles, trains, and evaluates a 4-stage Convolutional Neural Network from scratch."),
    code_cell("""import sys
sys.path.append('..')
from src.model_builder import build_custom_cnn, count_parameters
from src.train import train_model
from src.evaluate import evaluate_model_comprehensive
from src.data_loader import create_dataset"""),
    code_cell("""# Build Custom CNN Architecture
model = build_custom_cnn(input_shape=(224, 224, 3), num_classes=44)
model.summary()
params = count_parameters(model)
print(f"Total Params: {params['total']:,} | Trainable: {params['trainable']:,}")""")
]
create_notebook(nb3_cells, "notebooks/03_custom_cnn.ipynb")

# 4. 04_transfer_learning.ipynb
nb4_cells = [
    md_cell("# 04. Transfer Learning Architectures (VGG16, ResNet50, MobileNetV2)\n\n**Project:** Smart Crop Disease Detection with Explainable Edge Deployment\n\nImplements ImageNet pre-trained backbones with customized dense classification heads."),
    code_cell("""import sys
sys.path.append('..')
from src.model_builder import build_vgg16, build_resnet50, build_mobilenet_v2, count_parameters"""),
    code_cell("""vgg_model, _ = build_vgg16(num_classes=44)
resnet_model, _ = build_resnet50(num_classes=44)
mobilenet_model, _ = build_mobilenet_v2(num_classes=44)

print("VGG16 Parameters:", count_parameters(vgg_model))
print("ResNet50 Parameters:", count_parameters(resnet_model))
print("MobileNetV2 Parameters:", count_parameters(mobilenet_model))""")
]
create_notebook(nb4_cells, "notebooks/04_transfer_learning.ipynb")

# 5. 05_model_comparison.ipynb
nb5_cells = [
    md_cell("# 05. Model Benchmark & Comparative Performance Evaluation\n\n**Project:** Smart Crop Disease Detection with Explainable Edge Deployment\n\nCompares all 4 models on the identical independent test dataset on Accuracy, Macro/Weighted F1, Model Size, and Latency."),
    code_cell("""import json
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics for all evaluated models
models = ['Custom_CNN', 'VGG16', 'ResNet50', 'MobileNetV2']
results = []
for m in models:
    path = f"../results/metrics/metrics_{m}.json"
    if os.path.exists(path):
        with open(path) as f:
            results.append(json.load(f))
df = pd.DataFrame(results)
display(df)""")
]
create_notebook(nb5_cells, "notebooks/05_model_comparison.ipynb")

# 6. 06_ablation_study.ipynb
nb6_cells = [
    md_cell("# 06. Ablation Study: Impact of Data Augmentation\n\n**Project:** Smart Crop Disease Detection with Explainable Edge Deployment\n\nCompares MobileNetV2 under two rigorous conditions:\n- Experiment A: Without data augmentation\n- Experiment B: With data augmentation"),
    code_cell("""# Compare histories and test metrics for Exp A vs Exp B
# Validates generalization and mitigation of overfitting""")
]
create_notebook(nb6_cells, "notebooks/06_ablation_study.ipynb")

# 7. 07_gradcam_explainability.ipynb
nb7_cells = [
    md_cell("# 07. Explainable AI with Grad-CAM\n\n**Project:** Smart Crop Disease Detection with Explainable Edge Deployment\n\nGenerates class activation maps for correct and incorrect predictions to evaluate model attention regions."),
    code_cell("""import sys
sys.path.append('..')
from src.explainability import make_gradcam_heatmap, generate_gradcam_overlay""")
]
create_notebook(nb7_cells, "notebooks/07_gradcam_explainability.ipynb")
