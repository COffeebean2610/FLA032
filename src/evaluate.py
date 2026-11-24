import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import tensorflow as tf

def evaluate_model_comprehensive(model, test_dataset, class_names, model_name="Model", model_path=None, training_time_sec=0.0):
    """
    Computes all standard academic evaluation metrics on the independent test dataset:
    Accuracy, Macro/Weighted Precision, Recall, F1, Confusion Matrix, Latency, Throughput, and Model Size.
    """
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    y_true = []
    y_pred_probs = []
    
    print(f"\nEvaluating {model_name} on independent test set...")
    for images, labels in test_dataset:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(probs)
        
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Core Classification Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class Classification Report
    clf_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    with open(f"results/metrics/classification_report_{model_name}.json", "w") as f:
        json.dump(clf_report, f, indent=2)
        
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    np.save(f"results/metrics/confusion_matrix_{model_name}.npy", cm)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: {model_name} (Acc: {acc*100:.2f}%)", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_plot_path = f"results/figures/confusion_matrix_{model_name}.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    
    # Inference Latency Benchmark (100 repetitions on single images)
    sample_img = np.expand_dims(images.numpy()[0], axis=0) # batch size 1
    # Warmup
    for _ in range(10):
        _ = model(sample_img, training=False)
        
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model(sample_img, training=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # in ms
        
    mean_latency_ms = float(np.mean(latencies))
    std_latency_ms = float(np.std(latencies))
    fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0
    
    # Model Size on Disk
    model_size_mb = 0.0
    if model_path and os.path.exists(model_path):
        model_size_mb = os.path.getsize(model_path) / (1024.0 * 1024.0)
        
    # Parameter counts
    total_params = int(model.count_params())
    trainable_params = int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    
    metrics = {
        "model_name": model_name,
        "accuracy": float(acc),
        "macro_precision": float(macro_prec),
        "weighted_precision": float(weighted_prec),
        "macro_recall": float(macro_rec),
        "weighted_recall": float(weighted_rec),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "inference_latency_ms": mean_latency_ms,
        "inference_latency_std_ms": std_latency_ms,
        "throughput_fps": float(fps),
        "model_size_mb": float(model_size_mb),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "training_time_sec": float(training_time_sec),
        "confusion_matrix_plot": cm_plot_path
    }
    
    with open(f"results/metrics/metrics_{model_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Results for {model_name}:")
    print(f"  Accuracy:         {acc*100:.2f}%")
    print(f"  Macro F1-Score:   {macro_f1:.4f}")
    print(f"  Weighted F1:      {weighted_f1:.4f}")
    print(f"  Inference Time:   {mean_latency_ms:.2f} ms/image ({fps:.1f} FPS)")
    print(f"  Model Size:       {model_size_mb:.2f} MB")
    print(f"  Total Parameters: {total_params:,}")
    
    return metrics, y_true, y_pred, y_pred_probs
