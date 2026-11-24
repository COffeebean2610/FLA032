import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.data_loader import get_class_names, get_image_paths_and_labels
from src.preprocessing import check_corrupted_images, check_duplicates, plot_class_distribution

def main():
    print("=" * 70)
    print("PHASE 1 & 2: DATASET INVESTIGATION & PREPARATION")
    print("=" * 70)
    
    dataset_dir = os.path.join("dataset", "image data")
    splits = ["train", "validation", "test"]
    
    # 1. Class Discovery
    class_names = get_class_names(dataset_dir)
    print(f"Total Unique Classes Discovered: {len(class_names)}")
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
    print("Class names saved to: results/metrics/class_names.json")
    
    # 2. Count images per class per split
    summary_dict = {}
    for c in class_names:
        summary_dict[c] = {"train": 0, "val": 0, "test": 0, "total": 0}
        
    split_totals = {}
    for s in splits:
        paths, labels, _ = get_image_paths_and_labels(dataset_dir, split=s)
        split_totals[s] = len(paths)
        for idx in labels:
            c = class_names[idx]
            key = "val" if s == "validation" else s
            summary_dict[c][key] += 1
            summary_dict[c]["total"] += 1
            
    print(f"\nSplit Distribution:")
    for s, count in split_totals.items():
        pct = (count / sum(split_totals.values())) * 100.0
        print(f"  {s:<12}: {count:>6} images ({pct:.2f}%)")
    print(f"  {'TOTAL':<12}: {sum(split_totals.values()):>6} images (100.00%)")
    
    # Save dataset stats
    stats = {
        "dataset_name": "Leaf Disease Dataset Combination",
        "total_classes": len(class_names),
        "split_counts": split_totals,
        "total_images": sum(split_totals.values()),
        "classes": summary_dict
    }
    with open("results/metrics/dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Dataset statistics saved to: results/metrics/dataset_statistics.json")
    
    # 3. Generate Class Distribution Bar Chart
    plot_class_distribution(summary_dict, output_path="results/figures/class_distribution.png")
    
    # 4. Check for Corrupted Images
    print("\nScanning dataset for corrupt images...")
    total_scanned, corrupted = check_corrupted_images(dataset_dir)
    print(f"Scan complete. Scanned: {total_scanned} files. Corrupted: {len(corrupted)}")
    
    print("\nPhase 1 & 2 complete!")

if __name__ == "__main__":
    main()
