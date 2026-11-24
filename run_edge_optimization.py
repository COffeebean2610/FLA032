import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from src.edge_optimization import (
    convert_to_tflite_fp32,
    convert_to_tflite_fp16,
    convert_to_tflite_int8,
    benchmark_tflite_model
)
from src.data_loader import get_class_names

def main():
    print("=" * 80)
    print("PHASE 7: EDGE DEPLOYMENT & TENSORFLOW LITE QUANTIZATION")
    print("=" * 80)
    
    keras_path = "models/best_model/best_model.keras"
    if not os.path.exists(keras_path):
        keras_path = "models/MobileNetV2.keras"
        
    print(f"Loading Keras deployment model: {keras_path}")
    model = tf.keras.models.load_model(keras_path, safe_mode=False)
    original_size_mb = os.path.getsize(keras_path) / (1024.0 * 1024.0)
    
    # 1. Representative Dataset Generator for INT8 Calibration
    def representative_dataset():
        # Use small subset of test/val images
        test_dir = os.path.join("dataset", "image data", "test")
        sample_count = 0
        for root, _, files in os.walk(test_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fp = os.path.join(root, f)
                    img_bytes = tf.io.read_file(fp)
                    img = tf.image.decode_jpeg(img_bytes, channels=3)
                    img = tf.image.resize(img, (224, 224))
                    img = tf.cast(img, tf.float32) / 255.0
                    yield [tf.expand_dims(img, axis=0)]
                    sample_count += 1
                    if sample_count >= 100:
                        return
                        
    # 2. Exports
    print("\n1. Converting to Float32 TFLite...")
    fp32_path, fp32_size = convert_to_tflite_fp32(model)
    
    print("\n2. Converting to Float16 Quantized TFLite...")
    fp16_path, fp16_size = convert_to_tflite_fp16(model)
    
    print("\n3. Converting to Integer INT8 Quantized TFLite...")
    int8_path, int8_size = convert_to_tflite_int8(model, representative_dataset)
    
    # 3. Prepare Benchmark Evaluation Data (100 test images)
    print("\nPreparing test images for TFLite inference benchmarking...")
    test_imgs = []
    test_lbls = []
    class_names = get_class_names("dataset/image data")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for root, _, files in os.walk(os.path.join("dataset", "image data", "test")):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                parts = root.replace("\\", "/").split("/")
                # crop is parts[-2], disease is parts[-1]
                if len(parts) >= 2:
                    cname = f"{parts[-2]}___{parts[-1]}"
                    if cname in class_to_idx:
                        fp = os.path.join(root, f)
                        img_bytes = tf.io.read_file(fp)
                        img = tf.image.decode_jpeg(img_bytes, channels=3)
                        img = tf.image.resize(img, (224, 224))
                        img = tf.cast(img, tf.float32) / 255.0
                        test_imgs.append(img.numpy())
                        test_lbls.append(class_to_idx[cname])
                        if len(test_imgs) >= 120:
                            break
        if len(test_imgs) >= 120:
            break
            
    test_imgs = np.array(test_imgs)
    test_lbls = np.array(test_lbls)
    
    # 4. Benchmark each model
    print("\nBenchmarking TFLite Models on CPU...")
    benchmarks = []
    
    # Baseline Keras benchmark
    t0 = tf.timestamp()
    sample = test_imgs[0:1]
    for _ in range(10): _ = model(sample, training=False)
    keras_latencies = []
    for _ in range(100):
        t_start = tf.timestamp()
        _ = model(sample, training=False)
        t_end = tf.timestamp()
        keras_latencies.append((float(t_end - t_start)) * 1000.0)
    keras_latency = float(np.mean(keras_latencies))
    keras_preds = np.argmax(model.predict(test_imgs, verbose=0), axis=1)
    keras_acc = float(np.mean(keras_preds == test_lbls))
    
    benchmarks.append({
        "Model Format": "Keras FP32 Baseline",
        "Model Size (MB)": original_size_mb,
        "Compression": "1.00x",
        "Mean Latency (ms)": keras_latency,
        "Throughput (FPS)": 1000.0 / keras_latency,
        "Accuracy": keras_acc
    })
    
    # Benchmark TFLite models
    for name, p in [("TFLite Float32", fp32_path), ("TFLite Float16", fp16_path), ("TFLite INT8", int8_path)]:
        b = benchmark_tflite_model(p, test_imgs, test_lbls, num_runs=100)
        compression = original_size_mb / b['size_mb']
        benchmarks.append({
            "Model Format": name,
            "Model Size (MB)": b['size_mb'],
            "Compression": f"{compression:.2f}x",
            "Mean Latency (ms)": b['mean_latency_ms'],
            "Throughput (FPS)": b['throughput_fps'],
            "Accuracy": b['accuracy']
        })
        
    df_edge = pd.DataFrame(benchmarks)
    os.makedirs("results/comparison", exist_ok=True)
    df_edge.to_csv("results/comparison/edge_optimization_table.csv", index=False)
    with open("results/comparison/edge_optimization_table.json", "w") as f:
        json.dump(benchmarks, f, indent=2)
        
    print("\nEdge Deployment Optimization Summary:")
    print(df_edge.to_string(index=False))
    
    # 5. Plot Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    formats = [b['Model Format'] for b in benchmarks]
    sizes = [b['Model Size (MB)'] for b in benchmarks]
    latencies = [b['Mean Latency (ms)'] for b in benchmarks]
    
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Size Comparison
    ax1.bar(formats, sizes, color=palette)
    ax1.set_title("Model Footprint Across Quantization Formats (MB)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Size on Disk (MB)")
    ax1.tick_params(axis='x', rotation=20)
    for i, v in enumerate(sizes):
        ax1.text(i, v + (max(sizes)*0.02), f"{v:.2f} MB", ha='center', fontweight='bold', fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Latency Comparison
    ax2.bar(formats, latencies, color=palette)
    ax2.set_title("Inference Latency per Leaf Image (ms)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Latency (ms)")
    ax2.tick_params(axis='x', rotation=20)
    for i, v in enumerate(latencies):
        ax2.text(i, v + (max(latencies)*0.02), f"{v:.1f} ms", ha='center', fontweight='bold', fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("results/figures/edge_quantization_comparison.png", dpi=300)
    plt.close()
    print("Edge quantization figure saved: results/figures/edge_quantization_comparison.png")
    
    print("\nPhase 7 Edge Optimization completed successfully!")

if __name__ == "__main__":
    main()
