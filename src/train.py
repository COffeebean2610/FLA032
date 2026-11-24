import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_training_history(history, model_name="Model", output_dir="results/figures"):
    """
    Plots and saves training and validation accuracy and loss curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Plot
    ax1.plot(epochs, history.history['accuracy'], 'bo-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, history.history['val_accuracy'], 'ro-', label='Validation Accuracy', linewidth=2)
    ax1.set_title(f'{model_name} - Accuracy Trajectory', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=10)
    ax1.set_ylabel('Accuracy', fontsize=10)
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Loss Plot
    ax1_loss = ax2.plot(epochs, history.history['loss'], 'bo-', label='Training Loss', linewidth=2)
    ax2_loss = ax2.plot(epochs, history.history['val_loss'], 'ro-', label='Validation Loss', linewidth=2)
    ax2.set_title(f'{model_name} - Loss Trajectory', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=10)
    ax2.set_ylabel('Loss', fontsize=10)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"learning_curves_{model_name}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {plot_path}")
    return plot_path

def train_model(model, 
                train_ds, 
                val_ds, 
                model_name="Model", 
                epochs=10, 
                learning_rate=1e-3, 
                class_weights=None,
                checkpoint_dir="models"):
    """
    Executes disciplined model training with EarlyStopping, Learning Rate Scheduling, and Model Checkpointing.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}.keras")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print(f"\n{'='*70}")
    print(f"Starting Training: {model_name} (Max Epochs: {epochs}, LR: {learning_rate})")
    print(f"{'='*70}")
    
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60.0:.2f} minutes).")
    
    # Save training curves
    plot_training_history(history, model_name=model_name)
    
    # Save training history json
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_dict['training_time_sec'] = training_time
    with open(f"results/metrics/history_{model_name}.json", "w") as f:
        json.dump(history_dict, f, indent=2)
        
    return model, history, training_time, checkpoint_path
