import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf

from src.data_loader import get_class_names, parse_image
from src.explainability import find_last_conv_layer, make_gradcam_heatmap, generate_gradcam_overlay, visualize_and_save_gradcam_cases

def main():
    print("=" * 80)
    print("PHASE 6: EXPLAINABLE AI (GRAD-CAM) GENERATION")
    print("=" * 80)
    
    class_names_path = "results/metrics/class_names.json"
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            class_names = json.load(f)
    else:
        class_names = get_class_names("dataset/image data")
        
    model_path = "models/best_model/best_model.keras"
    if not os.path.exists(model_path):
        model_path = "models/MobileNetV2.keras"
        
    print(f"Loading trained deployment model from: {model_path}")
    model = tf.keras.models.load_model(model_path, safe_mode=False)
    
    # Identify target test images from diverse crops: Grape, Tomato, Rice, Potato, Pepper, Apple
    target_rel_paths = [
        ("dataset/image data/test/grape/black rot", "grape___black rot"),
        ("dataset/image data/test/tomato/early blight", "tomato___early blight"),
        ("dataset/image data/test/Rice/BrownSpot", "Rice___BrownSpot"),
        ("dataset/image data/test/potato/late blight", "potato___late blight"),
        ("dataset/image data/test/pepper, bell/bacterial spot", "pepper, bell___bacterial spot"),
        ("dataset/image data/test/apple/apple scab", "apple___apple scab")
    ]
    
    test_samples = []
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for folder, cname in target_rel_paths:
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                filepath = os.path.join(folder, files[0])
                pil_img = Image.open(filepath).convert('RGB').resize((224, 224))
                original_rgb = np.array(pil_img)
                img_array = original_rgb.astype(np.float32) / 255.0
                img_tensor = np.expand_dims(img_array, axis=0)
                
                true_label_idx = class_to_idx[cname]
                test_samples.append({
                    "tensor": img_tensor,
                    "original_rgb": original_rgb,
                    "true_label": true_label_idx,
                    "filepath": filepath
                })
                
    print(f"Selected {len(test_samples)} representative test samples for Grad-CAM analysis.")
    
    # Generate Grad-CAM heatmaps and multi-panel figures
    cases = visualize_and_save_gradcam_cases(
        model, test_samples, class_names, output_dir="results/figures/gradcam"
    )
    
    with open("results/figures/gradcam/gradcam_summary.json", "w") as f:
        json.dump(cases, f, indent=2)
        
    print(f"\nGrad-CAM generation complete! {len(cases)} diagnostic visual cases stored.")

if __name__ == "__main__":
    main()
