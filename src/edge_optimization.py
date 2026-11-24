import os
import time
import json
import numpy as np
import tensorflow as tf

def convert_to_tflite_fp32(keras_model, output_path="models/tflite/model_fp32.tflite"):
    """
    Converts a Keras model to unquantized Float32 TensorFlow Lite format.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(output_path) / (1024.0 * 1024.0)
    print(f"FP32 TFLite model exported: {output_path} ({size_mb:.2f} MB)")
    return output_path, size_mb

def convert_to_tflite_fp16(keras_model, output_path="models/tflite/model_fp16.tflite"):
    """
    Converts a Keras model to Float16 quantized TensorFlow Lite format.
    Reduces weight precision by 50% with near-zero loss in classification accuracy.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(output_path) / (1024.0 * 1024.0)
    print(f"FP16 Quantized TFLite model exported: {output_path} ({size_mb:.2f} MB)")
    return output_path, size_mb

def convert_to_tflite_int8(keras_model, representative_dataset_gen, output_path="models/tflite/model_int8.tflite"):
    """
    Converts a Keras model to Full Integer (INT8) quantized TensorFlow Lite format using representative calibration data.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    # Ensure ops support INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(output_path) / (1024.0 * 1024.0)
    print(f"INT8 Quantized TFLite model exported: {output_path} ({size_mb:.2f} MB)")
    return output_path, size_mb

def benchmark_tflite_model(tflite_path, test_images, test_labels, num_runs=100):
    """
    Evaluates inference latency, throughput (FPS), and classification accuracy of a TFLite model using TFLite Interpreter.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    # Latency benchmark on single image
    sample_img = test_images[0:1].astype(input_dtype)
    
    # Handle int8 quantization scaling if needed
    if input_dtype == np.int8 or input_dtype == np.uint8:
        scale, zero_point = input_details[0]['quantization']
        if scale > 0:
            sample_img = (sample_img / scale + zero_point).astype(input_dtype)
            
    interpreter.set_tensor(input_details[0]['index'], sample_img)
    
    # Warmup
    for _ in range(10):
        interpreter.invoke()
        
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], sample_img)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # in ms
        
    mean_latency = float(np.mean(latencies))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0
    
    # Accuracy evaluation on test subset
    correct = 0
    total = len(test_images)
    
    for i in range(total):
        img = test_images[i:i+1].astype(input_dtype)
        if input_dtype == np.int8 or input_dtype == np.uint8:
            scale, zero_point = input_details[0]['quantization']
            if scale > 0:
                img = (img / scale + zero_point).astype(input_dtype)
                
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        pred = np.argmax(out[0])
        if pred == test_labels[i]:
            correct += 1
            
    accuracy = float(correct / total) if total > 0 else 0.0
    size_mb = os.path.getsize(tflite_path) / (1024.0 * 1024.0)
    
    return {
        "model_file": os.path.basename(tflite_path),
        "size_mb": size_mb,
        "mean_latency_ms": mean_latency,
        "throughput_fps": fps,
        "accuracy": accuracy,
        "eval_samples": total
    }
