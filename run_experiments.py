import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Set strict random seeds for academic reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from src.data_loader import get_class_names, parse_image, get_augmentation_layer
from src.model_builder import build_custom_cnn, build_vgg16, build_resnet50, build_mobilenet_v2
from src.train import train_model
from src.evaluate import evaluate_model_comprehensive

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')

def get_stratified_file_list(dataset_dir="dataset/image data", split="train", max_per_class=70, seed=42):
    """
    Constructs a stratified, balanced subset of image paths across all 44 classes.
    """
    class_names = get_class_names(dataset_dir)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    split_dir = os.path.join(dataset_dir, split)
    
    rng = np.random.RandomState(seed)
    selected_paths = []
    selected_labels = []
    
    for crop in sorted(os.listdir(split_dir)):
        crop_path = os.path.join(split_dir, crop)
        if os.path.isdir(crop_path):
            for disease in sorted(os.listdir(crop_path)):
                disease_path = os.path.join(crop_path, disease)
                if os.path.isdir(disease_path):
                    class_name = f"{crop}___{disease}"
                    if class_name in class_to_idx:
                        label_idx = class_to_idx[class_name]
                        files = [os.path.join(disease_path, f) for f in os.listdir(disease_path) if f.endswith(IMAGE_EXTENSIONS)]
                        rng.shuffle(files)
                        chosen = files[:max_per_class] if max_per_class is not None else files
                        selected_paths.extend(chosen)
                        selected_labels.extend([label_idx] * len(chosen))
                        
    combined = list(zip(selected_paths, selected_labels))
    rng.shuffle(combined)
    paths, labels = zip(*combined)
    return np.array(paths), np.array(labels), class_names

def build_tf_dataset(paths, labels, batch_size=32, augment=False, shuffle=True, seed=42):
    """
    Constructs an optimized tf.data.Dataset from paths and labels.
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: parse_image(p, l, img_size=(224, 224)), num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        aug_layer = get_augmentation_layer()
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def main():
    print("=" * 80)
    print("STARTING SYSTEMATIC BENCHMARK & EXPERIMENTAL EVALUATION")
    print("Student: DAHIPHALE SAMRADNI ASHOK (ID: A:32)")
    print("=" * 80)
    
    dataset_dir = os.path.join("dataset", "image data")
    
    # 1. Stratified Data Selection (Ensures balanced representation across all 44 classes)
    print("\nPreparing stratified subsets across all 44 classes...")
    train_paths, train_labels, class_names = get_stratified_file_list(dataset_dir, "train", max_per_class=65, seed=42)
    val_paths, val_labels, _ = get_stratified_file_list(dataset_dir, "validation", max_per_class=15, seed=42)
    test_paths, test_labels, _ = get_stratified_file_list(dataset_dir, "test", max_per_class=15, seed=42)
    
    print(f"Stratified Training Samples:   {len(train_paths):>5} images across {len(class_names)} classes")
    print(f"Stratified Validation Samples: {len(val_paths):>5} images across {len(class_names)} classes")
    print(f"Stratified Testing Samples:    {len(test_paths):>5} images across {len(class_names)} classes")
    
    # Cache class names
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
        
    # Datasets
    batch_size = 64
    train_ds_aug = build_tf_dataset(train_paths, train_labels, batch_size=batch_size, augment=True)
    train_ds_noaug = build_tf_dataset(train_paths, train_labels, batch_size=batch_size, augment=False)
    val_ds = build_tf_dataset(val_paths, val_labels, batch_size=batch_size, augment=False, shuffle=False)
    test_ds = build_tf_dataset(test_paths, test_labels, batch_size=batch_size, augment=False, shuffle=False)
    
    epochs = 5
    results = []
    
    # =========================================================================
    # MODEL 1: Custom CNN Baseline
    # =========================================================================
    print("\n" + "#" * 80)
    print("EXPERIMENT 1: CUSTOM CNN BASELINE")
    print("#" * 80)
    cnn_metrics_path = "results/metrics/metrics_Custom_CNN.json"
    if os.path.exists(cnn_metrics_path):
        print("Found existing Custom_CNN evaluation metrics, loading...")
        with open(cnn_metrics_path) as f:
            cnn_metrics = json.load(f)
    else:
        cnn_model = build_custom_cnn(input_shape=(224, 224, 3), num_classes=len(class_names))
        cnn_model, cnn_hist, cnn_time, cnn_path = train_model(
            cnn_model, train_ds_aug, val_ds, model_name="Custom_CNN", epochs=epochs, learning_rate=1e-3
        )
        cnn_metrics, _, _, _ = evaluate_model_comprehensive(
            cnn_model, test_ds, class_names, model_name="Custom_CNN", model_path=cnn_path, training_time_sec=cnn_time
        )
    results.append(cnn_metrics)
    
    # =========================================================================
    # MODEL 2: VGG16 Transfer Learning
    # =========================================================================
    print("\n" + "#" * 80)
    print("EXPERIMENT 2: VGG16 TRANSFER LEARNING")
    print("#" * 80)
    vgg_model, _ = build_vgg16(input_shape=(224, 224, 3), num_classes=len(class_names), freeze_backbone=True)
    vgg_model, vgg_hist, vgg_time, vgg_path = train_model(
        vgg_model, train_ds_aug, val_ds, model_name="VGG16", epochs=epochs, learning_rate=1e-3
    )
    vgg_metrics, _, _, _ = evaluate_model_comprehensive(
        vgg_model, test_ds, class_names, model_name="VGG16", model_path=vgg_path, training_time_sec=vgg_time
    )
    results.append(vgg_metrics)
    
    # =========================================================================
    # MODEL 3: ResNet50 Transfer Learning
    # =========================================================================
    print("\n" + "#" * 80)
    print("EXPERIMENT 3: RESNET50 TRANSFER LEARNING")
    print("#" * 80)
    resnet_model, _ = build_resnet50(input_shape=(224, 224, 3), num_classes=len(class_names), freeze_backbone=True)
    resnet_model, resnet_hist, resnet_time, resnet_path = train_model(
        resnet_model, train_ds_aug, val_ds, model_name="ResNet50", epochs=epochs, learning_rate=1e-3
    )
    resnet_metrics, _, _, _ = evaluate_model_comprehensive(
        resnet_model, test_ds, class_names, model_name="ResNet50", model_path=resnet_path, training_time_sec=resnet_time
    )
    results.append(resnet_metrics)
    
    # =========================================================================
    # MODEL 4: MobileNetV2 Transfer Learning (Edge Optimized, With Augmentation)
    # =========================================================================
    print("\n" + "#" * 80)
    print("EXPERIMENT 4: MOBILENETV2 (WITH DATA AUGMENTATION)")
    print("#" * 80)
    mobilenet_model, mobilenet_base = build_mobilenet_v2(input_shape=(224, 224, 3), num_classes=len(class_names), freeze_backbone=True)
    mobilenet_model, mobilenet_hist, mobilenet_time, mobilenet_path = train_model(
        mobilenet_model, train_ds_aug, val_ds, model_name="MobileNetV2", epochs=epochs, learning_rate=1e-3
    )
    mobilenet_metrics, _, _, _ = evaluate_model_comprehensive(
        mobilenet_model, test_ds, class_names, model_name="MobileNetV2", model_path=mobilenet_path, training_time_sec=mobilenet_time
    )
    results.append(mobilenet_metrics)
    
    # =========================================================================
    # PHASE 5: ABLATION STUDY - MobileNetV2 Without Augmentation
    # =========================================================================
    print("\n" + "#" * 80)
    print("EXPERIMENT 5 (ABLATION): MOBILENETV2 WITHOUT DATA AUGMENTATION")
    print("#" * 80)
    mobilenet_noaug, _ = build_mobilenet_v2(input_shape=(224, 224, 3), num_classes=len(class_names), freeze_backbone=True)
    mobilenet_noaug, noaug_hist, noaug_time, noaug_path = train_model(
        mobilenet_noaug, train_ds_noaug, val_ds, model_name="MobileNetV2_NoAug", epochs=epochs, learning_rate=1e-3
    )
    noaug_metrics, _, _, _ = evaluate_model_comprehensive(
        mobilenet_noaug, test_ds, class_names, model_name="MobileNetV2_NoAug", model_path=noaug_path, training_time_sec=noaug_time
    )
    
    # Save Ablation Comparison
    ablation_data = [
        {
            "Condition": "Experiment A (Without Augmentation)",
            "Accuracy": noaug_metrics['accuracy'],
            "Macro_F1": noaug_metrics['macro_f1'],
            "Weighted_F1": noaug_metrics['weighted_f1'],
            "Final_Train_Loss": float(noaug_hist.history['loss'][-1]),
            "Final_Val_Loss": float(noaug_hist.history['val_loss'][-1]),
            "Overfitting_Gap": float(noaug_hist.history['val_loss'][-1] - noaug_hist.history['loss'][-1])
        },
        {
            "Condition": "Experiment B (With Augmentation)",
            "Accuracy": mobilenet_metrics['accuracy'],
            "Macro_F1": mobilenet_metrics['macro_f1'],
            "Weighted_F1": mobilenet_metrics['weighted_f1'],
            "Final_Train_Loss": float(mobilenet_hist.history['loss'][-1]),
            "Final_Val_Loss": float(mobilenet_hist.history['val_loss'][-1]),
            "Overfitting_Gap": float(mobilenet_hist.history['val_loss'][-1] - mobilenet_hist.history['loss'][-1])
        }
    ]
    with open("results/comparison/ablation_study.json", "w") as f:
        json.dump(ablation_data, f, indent=2)
    pd.DataFrame(ablation_data).to_csv("results/comparison/ablation_study.csv", index=False)
    
    # Plot Ablation Comparison Chart
    plt.figure(figsize=(9, 5))
    metrics_names = ['Accuracy', 'Macro F1', 'Weighted F1']
    vals_noaug = [noaug_metrics['accuracy'], noaug_metrics['macro_f1'], noaug_metrics['weighted_f1']]
    vals_aug = [mobilenet_metrics['accuracy'], mobilenet_metrics['macro_f1'], mobilenet_metrics['weighted_f1']]
    x = np.arange(len(metrics_names))
    width = 0.35
    plt.bar(x - width/2, vals_noaug, width, label='Without Augmentation (Exp A)', color='#d9534f')
    plt.bar(x + width/2, vals_aug, width, label='With Augmentation (Exp B)', color='#5cb85c')
    plt.ylabel('Score (0.0 to 1.0)', fontsize=11, fontweight='bold')
    plt.title('Ablation Study: Impact of Data Augmentation on MobileNetV2', fontsize=12, fontweight='bold')
    plt.xticks(x, metrics_names, fontsize=10)
    plt.ylim(0, 1.05)
    for i in range(len(metrics_names)):
        plt.text(x[i] - width/2, vals_noaug[i] + 0.02, f"{vals_noaug[i]*100:.1f}%", ha='center', fontsize=9)
        plt.text(x[i] + width/2, vals_aug[i] + 0.02, f"{vals_aug[i]*100:.1f}%", ha='center', fontsize=9)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/figures/ablation_study.png", dpi=300)
    plt.close()
    print("Ablation study chart saved: results/figures/ablation_study.png")
    
    # =========================================================================
    # CONSOLIDATED COMPARISON TABLE & VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("CONSOLIDATING COMPARATIVE BENCHMARK TABLE")
    print("=" * 80)
    
    os.makedirs("results/comparison", exist_ok=True)
    df_comp = pd.DataFrame([
        {
            "Model": r['model_name'],
            "Accuracy": f"{r['accuracy']*100:.2f}%",
            "Precision": f"{r['weighted_precision']:.4f}",
            "Recall": f"{r['weighted_recall']:.4f}",
            "F1-score": f"{r['weighted_f1']:.4f}",
            "Macro F1": f"{r['macro_f1']:.4f}",
            "Model Size (MB)": f"{r['model_size_mb']:.2f}",
            "Inference Time (ms)": f"{r['inference_latency_ms']:.2f}",
            "Throughput (FPS)": f"{r['throughput_fps']:.1f}",
            "Total Parameters": f"{r['total_parameters']:,}",
            "Training Time (s)": f"{r['training_time_sec']:.1f}"
        }
        for r in results
    ])
    
    df_comp.to_csv("results/comparison/comparison_table.csv", index=False)
    with open("results/comparison/comparison_table.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nConsolidated Model Comparison Table:")
    print(df_comp.to_string(index=False))
    
    # Model Comparison Multi-Bar Chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    models_labels = [r['model_name'] for r in results]
    accs = [r['accuracy'] * 100 for r in results]
    f1s = [r['macro_f1'] for r in results]
    sizes = [r['model_size_mb'] for r in results]
    latencies = [r['inference_latency_ms'] for r in results]
    
    colors = ['#4a90e2', '#50e3c2', '#f5a623', '#7ed321']
    
    # 1. Accuracy
    ax1.bar(models_labels, accs, color=colors)
    ax1.set_title("Test Accuracy (%)", fontweight='bold')
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 105)
    for i, v in enumerate(accs):
        ax1.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 2. Macro F1
    ax2.bar(models_labels, f1s, color=colors)
    ax2.set_title("Macro F1-Score", fontweight='bold')
    ax2.set_ylabel("F1 Score")
    ax2.set_ylim(0, 1.05)
    for i, v in enumerate(f1s):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold', fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 3. Model Size (MB)
    ax3.bar(models_labels, sizes, color=colors)
    ax3.set_title("Model Footprint on Disk (MB)", fontweight='bold')
    ax3.set_ylabel("Size (MB)")
    for i, v in enumerate(sizes):
        ax3.text(i, v + (max(sizes)*0.02), f"{v:.1f} MB", ha='center', fontweight='bold', fontsize=9)
    ax3.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 4. Latency (ms)
    ax4.bar(models_labels, latencies, color=colors)
    ax4.set_title("Inference Latency per Image (ms)", fontweight='bold')
    ax4.set_ylabel("Latency (ms)")
    for i, v in enumerate(latencies):
        ax4.text(i, v + (max(latencies)*0.02), f"{v:.1f} ms", ha='center', fontweight='bold', fontsize=9)
    ax4.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("results/figures/model_comparison.png", dpi=300)
    plt.close()
    print("Comparison figure saved: results/figures/model_comparison.png")
    
    # Save Best Model to models/best_model/
    # MobileNetV2 offers the optimal balance between high accuracy, minimal latency, and tiny footprint
    os.makedirs("models/best_model", exist_ok=True)
    best_model_dest = "models/best_model/best_model.keras"
    mobilenet_model.save(best_model_dest)
    print(f"\nBest deployment model saved to: {best_model_dest}")
    
    print("\nALL TRAINING AND COMPARATIVE EXPERIMENTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
