import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def find_last_conv_layer(model):
    """
    Automatically discovers the nested backbone and name of the last convolutional layer.
    """
    # Check if the model has a nested base_model
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)) or 'conv' in sub_layer.name.lower():
                    return layer, sub_layer.name
        elif isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)) or 'conv' in layer.name.lower():
            return None, layer.name
            
    # Default fallback
    return None, "Conv_1"

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, base_model=None):
    """
    Computes Grad-CAM heatmap for a given input image tensor and target class index.
    """
    # If base_model is nested inside model, extract features through base_model
    if base_model is not None:
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
        # Create a model that maps base_model input to last_conv_layer and base_model output
        feature_extractor = tf.keras.Model(
            inputs=base_model.inputs,
            outputs=[last_conv_layer.output, base_model.output]
        )
        
        # We need the classification head from model
        # Find index of base_model in model.layers
        base_idx = -1
        for idx, l in enumerate(model.layers):
            if l == base_model:
                base_idx = idx
                break
                
        # Forward pass with GradientTape
        with tf.GradientTape() as tape:
            # Map img_array through operational layers before base_model (skip InputLayer at 0)
            x = tf.convert_to_tensor(img_array)
            for l in model.layers[1:base_idx]:
                x = l(x)
            conv_outputs, base_out = feature_extractor(x)
            tape.watch(conv_outputs)
            
            # Map base_out through remaining layers after base_model
            preds = base_out
            for l in model.layers[base_idx+1:]:
                # Disable dropout during saliency inference
                if isinstance(l, tf.keras.layers.Dropout):
                    preds = l(preds, training=False)
                else:
                    preds = l(preds)
                
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
            
        grads = tape.gradient(class_channel, conv_outputs)
        
    else:
        # Standard unnested model (e.g. Custom CNN)
        last_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
            
        grads = tape.gradient(class_channel, conv_outputs)
        
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    
    # Weight feature map channels by gradient importance
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ReLU on heatmap and normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
        
    return heatmap.numpy()

def generate_gradcam_overlay(original_img_rgb, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays Grad-CAM heatmap onto the original RGB image.
    original_img_rgb: np.ndarray, shape (H, W, 3), range [0, 255] uint8.
    """
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (original_img_rgb.shape[1], original_img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply colormap (OpenCV returns BGR)
    colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap onto original image
    superimposed = np.float32(colored_heatmap_rgb) * alpha + np.float32(original_img_rgb) * (1.0 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
    return heatmap_resized, superimposed

def visualize_and_save_gradcam_cases(model, test_samples, class_names, base_model=None, last_conv_layer_name=None, output_dir="results/figures/gradcam"):
    """
    Generates and saves comprehensive Grad-CAM analysis for sample test images.
    Visualizes original image, Grad-CAM heatmap, superimposed overlay, predicted class, and ground truth.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if last_conv_layer_name is None:
        nested_base, last_conv_layer_name = find_last_conv_layer(model)
        if base_model is None and nested_base is not None:
            base_model = nested_base
            
    case_results = []
    
    for i, sample in enumerate(test_samples):
        img_tensor = sample['tensor'] # (1, 224, 224, 3) normalized [0, 1]
        original_img = sample['original_rgb'] # (224, 224, 3) uint8 [0, 255]
        true_label_idx = sample['true_label']
        true_class_name = class_names[true_label_idx]
        
        preds = model.predict(img_tensor, verbose=0)[0]
        pred_label_idx = np.argmax(preds)
        pred_class_name = class_names[pred_label_idx]
        confidence = float(preds[pred_label_idx])
        is_correct = (pred_label_idx == true_label_idx)
        
        heatmap = make_gradcam_heatmap(
            img_tensor, model, 
            last_conv_layer_name=last_conv_layer_name, 
            pred_index=pred_label_idx, 
            base_model=base_model
        )
        
        heatmap_resized, overlay = generate_gradcam_overlay(original_img, heatmap)
        
        # Save individual multi-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_img)
        axes[0].set_title(f"Original Image\nGround Truth: {true_class_name}", fontsize=11)
        axes[0].axis('off')
        
        im1 = axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap\n(Salient Activation Map)", fontsize=11)
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        axes[2].imshow(overlay)
        status_str = "CORRECT" if is_correct else "INCORRECT"
        axes[2].set_title(f"Overlay [{status_str}]\nPred: {pred_class_name} ({confidence*100:.1f}%)", 
                          fontsize=11, color='green' if is_correct else 'red', fontweight='bold')
        axes[2].axis('off')
        
        filename = f"gradcam_case_{i+1}_{'correct' if is_correct else 'incorrect'}_{true_class_name.replace('___', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        
        case_results.append({
            "case_id": i + 1,
            "filename": filename,
            "filepath": filepath,
            "true_class": true_class_name,
            "pred_class": pred_class_name,
            "confidence": float(confidence),
            "is_correct": bool(is_correct)
        })
        print(f"Grad-CAM Case {i+1} saved: {filename} (True: {true_class_name} | Pred: {pred_class_name} [{confidence*100:.1f}%])")
        
    return case_results
