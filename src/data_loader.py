import os
import glob
import json
import numpy as np
import tensorflow as tf

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')

def get_class_names(dataset_dir="dataset/image data"):
    """
    Scans dataset directory and returns a sorted list of all unique 'Crop___Disease' classes.
    """
    train_dir = os.path.join(dataset_dir, "train")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")
    
    classes = []
    for crop in sorted(os.listdir(train_dir)):
        crop_path = os.path.join(train_dir, crop)
        if os.path.isdir(crop_path):
            for disease in sorted(os.listdir(crop_path)):
                disease_path = os.path.join(crop_path, disease)
                if os.path.isdir(disease_path):
                    classes.append(f"{crop}___{disease}")
    
    return sorted(list(set(classes)))

def get_image_paths_and_labels(dataset_dir="dataset/image data", split="train"):
    """
    Collects all image file paths and corresponding integer label indices for a given split.
    """
    class_names = get_class_names(dataset_dir)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    split_dir = os.path.join(dataset_dir, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    image_paths = []
    labels = []
    
    for crop in sorted(os.listdir(split_dir)):
        crop_path = os.path.join(split_dir, crop)
        if os.path.isdir(crop_path):
            for disease in sorted(os.listdir(crop_path)):
                disease_path = os.path.join(crop_path, disease)
                if os.path.isdir(disease_path):
                    class_name = f"{crop}___{disease}"
                    if class_name in class_to_idx:
                        label_idx = class_to_idx[class_name]
                        for fname in os.listdir(disease_path):
                            if fname.endswith(IMAGE_EXTENSIONS):
                                full_path = os.path.join(disease_path, fname)
                                image_paths.append(full_path)
                                labels.append(label_idx)
                                
    return np.array(image_paths), np.array(labels), class_names

def parse_image(path, label, img_size=(224, 224)):
    """
    Reads and decodes an image file into a normalized 224x224 RGB float32 tensor.
    """
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def get_augmentation_layer():
    """
    Returns Keras Sequential preprocessing layers for training data augmentation.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
    ], name="data_augmentation")

def create_dataset(dataset_dir="dataset/image data", 
                   split="train", 
                   batch_size=32, 
                   img_size=(224, 224), 
                   augment=False, 
                   shuffle=True, 
                   sample_fraction=None,
                   random_seed=42):
    """
    Constructs an optimized tf.data.Dataset pipeline.
    """
    paths, labels, class_names = get_image_paths_and_labels(dataset_dir, split)
    
    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        np.random.seed(random_seed)
        n_samples = int(len(paths) * sample_fraction)
        indices = np.random.choice(len(paths), size=n_samples, replace=False)
        paths = paths[indices]
        labels = labels[indices]
        
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 10000), seed=random_seed, reshuffle_each_iteration=True)
        
    ds = ds.map(lambda p, l: parse_image(p, l, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        aug_layer = get_augmentation_layer()
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return ds, len(paths), class_names
