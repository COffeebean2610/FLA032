import os
import hashlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

def check_corrupted_images(dataset_dir="dataset/image data"):
    """
    Scans the dataset directory to verify all image files can be opened and decoded without corruption.
    """
    corrupted = []
    total = 0
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')
    
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith(valid_exts):
                total += 1
                filepath = os.path.join(root, f)
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                except Exception as e:
                    corrupted.append((filepath, str(e)))
                    
    return total, corrupted

def compute_dataset_class_weights(labels):
    """
    Computes balanced class weights for training to handle class imbalance.
    """
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))

def plot_class_distribution(summary_dict, output_path="results/figures/class_distribution.png"):
    """
    Generates a horizontal bar chart displaying image counts across all classes and splits.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    classes = sorted(list(summary_dict.keys()))
    train_counts = [summary_dict[c].get('train', 0) for c in classes]
    val_counts = [summary_dict[c].get('val', 0) for c in classes]
    test_counts = [summary_dict[c].get('test', 0) for c in classes]
    
    fig, ax = plt.subplots(figsize=(12, 16))
    y = np.arange(len(classes))
    height = 0.25
    
    ax.barh(y - height, train_counts, height, label='Train', color='#2b5c8f')
    ax.barh(y, val_counts, height, label='Validation', color='#e28743')
    ax.barh(y + height, test_counts, height, label='Test', color='#2ca02c')
    
    ax.set_xlabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Class Distribution Across Splits (44 Classes)', fontsize=14, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=9)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Class distribution plot saved to: {output_path}")

def check_duplicates(dataset_dir="dataset/image data", sample_limit=5000):
    """
    Computes MD5 hashes for a subset of images to evaluate duplicate frequency.
    """
    hashes = {}
    duplicates = []
    scanned = 0
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')
    
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith(valid_exts):
                scanned += 1
                filepath = os.path.join(root, f)
                with open(filepath, 'rb') as fp:
                    file_hash = hashlib.md5(fp.read()).hexdigest()
                    if file_hash in hashes:
                        duplicates.append((filepath, hashes[file_hash]))
                    else:
                        hashes[file_hash] = filepath
                if sample_limit and scanned >= sample_limit:
                    break
        if sample_limit and scanned >= sample_limit:
            break
            
    return scanned, duplicates
