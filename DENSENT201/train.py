"""
DenseNet201 Trainer
- Phase 1: frozen base, train custom head
- Phase 2: unfreeze last 25% for fine-tuning
- Cosine LR decay, label smoothing, class balancing
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

IMG_SIZE = (224, 224)  # DenseNet201 standard input
BATCH_SIZE = 32
NUM_CLASSES = 4

# Phase 1 (head training)
PHASE1_EPOCHS = 15
PHASE1_LR = 1e-3

# Phase 2 (fine-tuning)
PHASE2_EPOCHS = 30
PHASE2_LR = 1e-5
UNFREEZE_FROM_PCT = 0.75  # Unfreeze top 25%

# Regularization
DROPOUT_RATE = 0.4
L2_REG = 1e-4
LABEL_SMOOTHING = 0.1

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

CLASS_NAMES = []

def build_model(trainable_base: bool = False) -> tuple:
    """Build DenseNet201 model with custom head."""
    base_model = tf.keras.applications.DenseNet201(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = trainable_base

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        512,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(L2_REG)
    )(x)
    x = layers.Dropout(DROPOUT_RATE / 2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model, base_model


def make_dataset(directory: str, shuffle: bool = False) -> tf.data.Dataset:
    """Create tf.data pipeline."""
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=42
    )

    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.densenet.preprocess_input(image)
        return image, label

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weights(train_dir):
    """Compute class weights for imbalanced data."""
    class_counts = [len(os.listdir(os.path.join(train_dir, c))) for c in CLASS_NAMES]
    class_weight = compute_class_weight(
        'balanced',
        classes=np.arange(NUM_CLASSES),
        y=np.repeat(np.arange(NUM_CLASSES), class_counts)
    )
    return dict(enumerate(class_weight))


def plot_training_history(history_p1, history_p2, output_path):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Phase 1 accuracy
    axes[0, 0].plot(history_p1.history['accuracy'], label='train', linewidth=2)
    axes[0, 0].plot(history_p1.history['val_accuracy'], label='val', linewidth=2)
    axes[0, 0].set_title('Phase 1: Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Phase 1 loss
    axes[0, 1].plot(history_p1.history['loss'], label='train', linewidth=2)
    axes[0, 1].plot(history_p1.history['val_loss'], label='val', linewidth=2)
    axes[0, 1].set_title('Phase 1: Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Phase 2 accuracy
    axes[1, 0].plot(history_p2.history['accuracy'], label='train', linewidth=2)
    axes[1, 0].plot(history_p2.history['val_accuracy'], label='val', linewidth=2)
    axes[1, 0].set_title('Phase 2: Accuracy', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Phase 2 loss
    axes[1, 1].plot(history_p2.history['loss'], label='train', linewidth=2)
    axes[1, 1].plot(history_p2.history['val_loss'], label='val', linewidth=2)
    axes[1, 1].set_title('Phase 2: Loss', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to {output_path}')


def train(train_dir, val_dir, output_dir):
    """Main training pipeline."""
    global CLASS_NAMES

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    CLASS_NAMES = sorted(os.listdir(train_dir))
    print(f'Classes: {CLASS_NAMES}')
    print(f'Num classes: {len(CLASS_NAMES)}')

    if not os.path.isdir(train_dir):
        raise ValueError(f'Train directory not found: {train_dir}')
    if not os.path.isdir(val_dir):
        raise ValueError(f'Validation directory not found: {val_dir}')

    print('\nImage counts:')
    for split, path in [('train', train_dir), ('validate', val_dir)]:
        try:
            total = sum(len(os.listdir(os.path.join(path, c))) for c in CLASS_NAMES)
            per_class = {c: len(os.listdir(os.path.join(path, c))) for c in CLASS_NAMES}
            print(f'  {split:<10}: {total} total  |  {per_class}')
        except Exception as e:
            print(f'  Error reading {split} directory: {e}')

    print('\nCreating data pipelines...')
    train_ds = make_dataset(train_dir, shuffle=True)
    val_ds = make_dataset(val_dir, shuffle=False)
    print(f'Train batches: {len(train_ds)} | Val batches: {len(val_ds)}')

    class_weight = compute_class_weights(train_dir)

    print('\nBuilding model...')
    model, base_model = build_model(trainable_base=False)
    print(f'Total params: {model.count_params():,}')

    best_model_path = os.path.join(output_dir, 'best_densenet201.keras')

    # ========================================================================
    # PHASE 1: Train Head Only
    # ========================================================================
    print('\n' + '='*60)
    print('PHASE 1: Training custom head (base frozen)')
    print('='*60)

    steps_per_epoch = len(train_ds)
    total_steps_p1 = PHASE1_EPOCHS * steps_per_epoch

    lr_schedule_p1 = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=PHASE1_LR,
        decay_steps=total_steps_p1,
        alpha=1e-6
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule_p1),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc')]
    )

    callbacks_p1 = [
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=7,
            restore_best_weights=True, verbose=1
        )
    ]

    history_p1 = model.fit(
        train_ds,
        epochs=PHASE1_EPOCHS,
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=callbacks_p1,
        verbose=1
    )

    print(f'\nPhase 1 best: {max(history_p1.history["val_accuracy"]):.4f}')

    # ========================================================================
    # PHASE 2: Fine-Tune
    # ========================================================================
    print('\n' + '='*60)
    print('PHASE 2: Fine-tuning (top 25% unfrozen)')
    print('='*60)

    base_model.trainable = True
    fine_tune_at = int(len(base_model.layers) * UNFREEZE_FROM_PCT)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    unfrozen = sum(1 for l in base_model.layers if l.trainable)
    print(f'Unfrozen layers: {unfrozen} / {len(base_model.layers)}')

    lr_schedule_p2 = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=PHASE2_LR,
        first_decay_steps=steps_per_epoch * 5,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-7
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule_p2),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc')]
    )

    callbacks_p2 = [
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=12,
            restore_best_weights=True, verbose=1
        )
    ]

    history_p2 = model.fit(
        train_ds,
        epochs=PHASE2_EPOCHS,
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=callbacks_p2,
        verbose=1
    )

    print(f'\nPhase 2 best: {max(history_p2.history["val_accuracy"]):.4f}')

    plot_training_history(history_p1, history_p2,
                          os.path.join(output_dir, 'training_curves.png'))

    with open(os.path.join(output_dir, 'class_names.json'), 'w') as f:
        json.dump(CLASS_NAMES, f)

    print(f'\n✓ Training complete!')
    print(f'  Model: {best_model_path}')
    print(f'  Classes: {os.path.join(output_dir, "class_names.json")}')


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DenseNet201')
    parser.add_argument('--train-dir', type=str, default='new dataset/train',
                        help='Path to training directory')
    parser.add_argument('--val-dir', type=str, default='new dataset/valid',
                        help='Path to validation directory')
    parser.add_argument('--output-dir', type=str, default='./models',
                        help='Output directory for model')

    args = parser.parse_args()

    print('\n' + '='*60)
    print('DenseNet201 Trainer')
    print('='*60)
    print(f'TensorFlow: {tf.__version__}')
    print(f'GPU: {tf.config.list_physical_devices("GPU")}')
    print(f'Train: {args.train_dir}')
    print(f'Val: {args.val_dir}')
    print(f'Output: {args.output_dir}')
    print('='*60 + '\n')

    train(args.train_dir, args.val_dir, args.output_dir)
