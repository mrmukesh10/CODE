"""
ResNet101 — Disease Detection Model Trainer (4 Classes)
Python 3.10 compatible - Training script

Classes: Dermatophilosis, FMD (Foot and Mouth Disease), Healthy, Lumpy Skin

This script trains a ResNet101 model using:
- Phase 1: frozen base, train only custom head
- Phase 2: unfreeze last 30% of base layers for fine-tuning
- Cosine LR decay, label smoothing, MixUp, class balancing
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
import argparse
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4

# Phase 1 (head training)
PHASE1_EPOCHS = 15
PHASE1_LR = 1e-3

# Phase 2 (fine-tuning)
PHASE2_EPOCHS = 35
PHASE2_LR = 1e-5
UNFREEZE_FROM_PCT = 0.70

# Regularization
DROPOUT_RATE = 0.4
L2_REG = 1e-4
LABEL_SMOOTHING = 0.1
USE_MIXUP = True
MIXUP_ALPHA = 0.2

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def build_model(trainable_base: bool = False) -> tuple:
    """Build ResNet101 model with custom head."""
    base_model = tf.keras.applications.ResNet101(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = trainable_base

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(x)
    x = layers.Dropout(DROPOUT_RATE / 2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model, base_model


def get_class_weights(train_dir, class_names):
    """Calculate class weights for imbalanced dataset."""
    class_counts = {c: 0 for c in class_names}
    for class_name in class_names:
        class_dir = os.path.join(train_dir, class_name)
        class_counts[class_name] = len([f for f in os.listdir(class_dir) 
                                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print('\nClass distribution:')
    for class_name, count in class_counts.items():
        print(f'  {class_name:<20}: {count:4d} images')
    
    # Compute class weights
    class_list = list(class_counts.keys())
    weights = compute_class_weight(
        'balanced',
        classes=np.arange(len(class_list)),
        y=np.repeat(np.arange(len(class_list)), list(class_counts.values()))
    )
    
    class_weight_dict = {i: w for i, w in enumerate(weights)}
    print('\nClass weights:')
    for i, class_name in enumerate(class_list):
        print(f'  {class_name:<20}: {class_weight_dict[i]:.4f}')
    
    return class_weight_dict


def mixup(x, y, alpha=0.2):
    """Apply MixUp augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = tf.shape(x)[0]
    index = tf.random.shuffle(tf.range(batch_size))
    
    x_mixed = lam * x + (1 - lam) * tf.gather(x, index)
    y_mixed = lam * y + (1 - lam) * tf.gather(y, index)
    
    return x_mixed, y_mixed


def train_phase(model, train_ds, val_ds, epochs, learning_rate, class_weight, phase_name):
    """Train model for one phase."""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
        ]
    )
    
    print(f'\n{"="*60}')
    print(f'{phase_name}')
    print(f'{"="*60}')
    print(f'Epochs: {epochs}')
    print(f'Learning Rate: {learning_rate}')
    print(f'Batch Size: {BATCH_SIZE}')
    
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def plot_history(history, output_dir):
    """Plot training history."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(output_path, dpi=100)
    print(f'\n✓ Training history saved: {output_path}')
    plt.close()


def main():
    """Main training function."""
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(description='Train ResNet101 model')
    parser.add_argument(
        '--train-dir',
        type=str,
        default=os.path.join(project_dir, 'new dataset', 'train'),
        help='Training directory'
    )
    parser.add_argument(
        '--val-dir',
        type=str,
        default=os.path.join(project_dir, 'new dataset', 'valid'),
        help='Validation directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=script_dir,
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Load class names
    class_names_path = os.path.join(project_dir, 'models', 'class_names.json')
    if not os.path.exists(class_names_path):
        print(f'ERROR: Class names file not found at {class_names_path}')
        return
    
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    print('\n' + '='*60)
    print('ResNet101 Disease Detection Model Training')
    print('='*60)
    print(f'\nClasses: {class_names}')
    print(f'Image Size: {IMG_SIZE}')
    print(f'Batch Size: {BATCH_SIZE}')
    
    # Create datasets
    print(f'\nLoading training data from: {args.train_dir}')
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.train_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    print(f'Loading validation data from: {args.val_dir}')
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.val_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Preprocessing
    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.resnet.preprocess_input(image)
        return image, label
    
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.cache()
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    # Get class weights
    print('\nCalculating class weights...')
    class_weight = get_class_weights(args.train_dir, class_names)
    
    # Build model
    print('\n' + '='*60)
    print('Building ResNet101 Model')
    print('='*60)
    model, base_model = build_model(trainable_base=False)
    model.summary()
    
    # Phase 1: Train head only
    print('\n' + '='*60)
    print('PHASE 1: Training Custom Head (Base Frozen)')
    print('='*60)
    history_phase1 = train_phase(
        model, train_ds, val_ds,
        epochs=PHASE1_EPOCHS,
        learning_rate=PHASE1_LR,
        class_weight=class_weight,
        phase_name='Phase 1: Head Training'
    )
    
    # Phase 2: Fine-tune base layers
    print('\n' + '='*60)
    print('PHASE 2: Fine-tuning Base Layers')
    print('='*60)
    
    # Unfreeze last 30% of base layers
    total_layers = len(base_model.layers)
    unfreeze_from = int(total_layers * UNFREEZE_FROM_PCT)
    
    for layer in base_model.layers[unfreeze_from:]:
        layer.trainable = True
    
    print(f'Unfreezing {total_layers - unfreeze_from}/{total_layers} base layers')
    
    model.summary()
    
    history_phase2 = train_phase(
        model, train_ds, val_ds,
        epochs=PHASE2_EPOCHS,
        learning_rate=PHASE2_LR,
        class_weight=class_weight,
        phase_name='Phase 2: Fine-tuning'
    )
    
    # Save model
    print('\n' + '='*60)
    print('Saving Model')
    print('='*60)
    
    model_path = os.path.join(args.output_dir, 'best_resnet101.keras')
    model.save(model_path)
    print(f'✓ Model saved: {model_path}')
    
    # Also save to models directory
    models_dir = os.path.join(project_dir, 'models')
    if os.path.exists(models_dir):
        models_model_path = os.path.join(models_dir, 'best_resnet101.keras')
        model.save(models_model_path)
        print(f'✓ Model saved: {models_model_path}')
    
    # Plot history
    plot_history(history_phase2, args.output_dir)
    
    print('\n' + '='*60)
    print('✓ Training Complete!')
    print('='*60)


if __name__ == '__main__':
    main()
