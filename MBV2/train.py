"""
MobileNetV2 Trainer
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

IMG_SIZE = (224, 224)  # Standard input size
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
    """Build MobileNetV2 model with custom head."""
    base_model = tf.keras.applications.MobileNetV2(
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
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
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


def train():
    """Main training pipeline."""
    global CLASS_NAMES
    
    parser = argparse.ArgumentParser(description='Train MobileNetV2')
    parser.add_argument('--train-dir', default=r'D:\project\new dataset\train', help='Training directory')
    parser.add_argument('--val-dir', default=r'D:\project\new dataset\valid', help='Validation directory')
    parser.add_argument('--output-dir', default=r'D:\project\mobilenetv2\results', help='Output directory')
    parser.add_argument('--models-dir', default=r'D:\project\models', help='Models directory')
    args = parser.parse_args()

    # Setup
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    
    CLASS_NAMES = ['dermatophilosis', 'fmd', 'healthy', 'lumpy skin']

    print('\n' + '='*70)
    print('  MobileNetV2 CATTLE DISEASE CLASSIFIER - TRAINING')
    print('='*70)
    print(f'Classes: {CLASS_NAMES}')
    print(f'Image size: {IMG_SIZE[0]}×{IMG_SIZE[1]} | Batch: {BATCH_SIZE}')
    print(f'Output dir: {args.output_dir}')

    # Datasets
    print('\n[1/5] Loading datasets...')
    train_ds = make_dataset(args.train_dir, shuffle=True)
    val_ds = make_dataset(args.val_dir, shuffle=False)
    print('✓ Datasets loaded')

    # Class weights
    print('[2/5] Computing class weights...')
    class_weight = compute_class_weights(args.train_dir)
    print(f'✓ Class weights: {class_weight}')

    # Build model
    print('[3/5] Building model...')
    model, base_model = build_model(trainable_base=False)
    print(f'✓ Model created: {model.count_params():,} parameters')

    # Phase 1: Head training
    print('\n[4a/5] Phase 1: Training head (frozen base)...')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )

    history_p1 = model.fit(
        train_ds,
        epochs=PHASE1_EPOCHS,
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
        ],
        verbose=1
    )
    print('✓ Phase 1 complete')

    # Phase 2: Fine-tuning
    print('\n[4b/5] Phase 2: Fine-tuning base layers...')
    total_layers = len(base_model.layers)
    unfreeze_from = int(total_layers * UNFREEZE_FROM_PCT)
    
    for layer in base_model.layers[unfreeze_from:]:
        layer.trainable = True
    
    print(f'Unfrozing {total_layers - unfreeze_from}/{total_layers} layers')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
    )

    history_p2 = model.fit(
        train_ds,
        epochs=PHASE2_EPOCHS,
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
        ],
        verbose=1
    )
    print('✓ Phase 2 complete')

    # Save model
    print('\n[5/5] Saving results...')
    model_path = os.path.join(args.models_dir, 'best_mobilenetv2.keras')
    model.save(model_path)
    print(f'✓ Model saved: {model_path}')

    # Save class names
    class_names_path = os.path.join(args.models_dir, 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(CLASS_NAMES, f, indent=2)
    print(f'✓ Class names saved')

    # Plot history
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history_p1, history_p2, plot_path)
    print(f'✓ Training history saved: {plot_path}')

    print('\n' + '='*70)
    print('  ✓ TRAINING COMPLETE!')
    print('='*70 + '\n')


if __name__ == '__main__':
    train()
"""
MobileNetV2 — Disease Detection Model Trainer (4 Classes) - PRODUCTION READY
Classes: Dermatophilosis, FMD, Healthy, Lumpy Skin
Confidence fix: Label Smoothing=0, Dropout=0.30, Temperature Scaling
"""

import os
import json
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import argparse
from pathlib import Path

# ── Mixed Precision Training (Disabled for CPU compatibility) ──────────────
# Disabled: Requires AVX-512 support. Use standard float32 instead.
# if True:
#     policy = tf.keras.mixed_precision.Policy('mixed_float16')
#     tf.keras.mixed_precision.set_global_policy(policy)

# ── Suppress all TF/Keras noise ─────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

# ============================================================================
# CONFIGURATION (OPTIMIZED)
# ============================================================================

IMG_SIZE          = (384, 384)  # Increased from 256x256 for better detail capture
BATCH_SIZE        = 16  # Reduced from 32 to prevent memory overflow with larger images
NUM_CLASSES       = 4
PHASE1_EPOCHS     = 20  # Increased from 15
PHASE1_LR         = 5e-4  # Decreased from 1e-3 (slower warmup)
PHASE2_EPOCHS     = 50  # Increased from 35 (more fine-tuning)
PHASE2_LR         = 5e-6  # Decreased from 1e-5 (conservative fine-tuning)
UNFREEZE_FROM_PCT = 0.80  # Increased from 0.70 (unfreeze more layers)
DROPOUT_RATE      = 0.40  # Increased from 0.30 for better regularization
L2_REG            = 1e-4
LABEL_SMOOTHING   = 0.0      # FIXED: Allows high confidence
TEMPERATURE_CALIBRATION = True
USE_MIXED_PRECISION = True  # NEW: Enable mixed precision training

# ============================================================================
# PRINT HELPERS
# ============================================================================

def section(title):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")

def ok(msg):    print(f"  ✓  {msg}")
def info(msg):  print(f"    {msg}")

# ============================================================================
# CLEAN EPOCH LOGGER
# ============================================================================

class CleanLogger(keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self._total = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ep   = epoch + 1
        lr   = float(self.model.optimizer.learning_rate)
        conf = logs.get('val_mean_confidence', 0)
        
        line = (f"  Epoch {ep:>3}/{self._total}  "
                f"acc {logs.get('accuracy', 0):.3f}  "
                f"val_acc {logs.get('val_accuracy', 0):.3f}  "
                f"loss {logs.get('loss', 0):.3f}  "
                f"val_loss {logs.get('val_loss', 0):.3f}  "
                f"lr {lr:.1e}  conf {conf:.3f}")
        print(line)

# ============================================================================
# CONFIDENCE CALLBACK
# ============================================================================

class MeanConfidenceCallback(keras.callbacks.Callback):
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        vals = np.concatenate([
            tf.reduce_max(self.model(imgs, training=False), axis=1).numpy()
            for imgs, _ in self.val_ds
        ])
        if logs is not None:
            logs['val_mean_confidence'] = float(vals.mean())

# ============================================================================
# MODEL
# ============================================================================

def build_model(trainable_base=False):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet'
    )
    base.trainable = trainable_base

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu',  # Increased from 256
                    kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(256, activation='relu',  # Increased from 128
                    kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)  # NEW: Additional BN layer
    x = layers.Dropout(DROPOUT_RATE * 0.75)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return keras.Model(inputs, outputs), base

# ============================================================================
# DATA
# ============================================================================

def get_class_weights(train_dir, class_names):
    counts = {}
    for name in class_names:
        d = os.path.join(train_dir, name)
        counts[name] = len([
            f for f in os.listdir(d)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    section("Dataset Distribution")
    total = sum(counts.values())
    for name, n in counts.items():
        bar = '█' * int(30 * n / total)
        info(f"{name:<22}  {n:>5} images  {bar}")

    weights = compute_class_weight(
        'balanced',
        classes=np.arange(len(class_names)),
        y=np.repeat(np.arange(len(class_names)), list(counts.values()))
    )
    return {i: w for i, w in enumerate(weights)}

def make_augmentation_layer():
    return keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.20),  # Increased from 0.15
        layers.RandomZoom(0.15),  # Increased from 0.10
        layers.RandomBrightness(0.20),  # Increased from 0.15
        layers.RandomContrast(0.20),  # Increased from 0.15
        layers.RandomTranslation(0.10, 0.10),  # NEW: Random shifts
        layers.GaussianNoise(0.02),  # NEW: Noise regularization
    ], name='augmentation')

# ============================================================================
# TRAINING
# ============================================================================

def train_phase(model, train_ds, val_ds, epochs, lr, class_weight, label, is_phase2=False):
    # Freeze BatchNormalization layers during fine-tuning for stability
    if is_phase2:
        for layer in model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0, beta_2=0.999),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[
            keras.metrics.CategoricalAccuracy(name='accuracy'),
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
        ]
    )

    section(label)
    info(f"Epochs        : {epochs}")
    info(f"Learning rate : {lr}")
    info(f"Label smooth  : {LABEL_SMOOTHING}")
    info(f"Dropout       : {DROPOUT_RATE}")
    info(f"Image size    : {IMG_SIZE[0]}×{IMG_SIZE[1]}")
    print()

    # Cosine decay learning rate scheduler
    def cosine_decay_lr(epoch):
        return lr * 0.5 * (1 + np.cos(np.pi * (epoch / epochs)))

    history = model.fit(
        train_ds, epochs=epochs,
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=[
            CleanLogger(epochs),
            MeanConfidenceCallback(val_ds),
            keras.callbacks.LearningRateScheduler(cosine_decay_lr, verbose=0),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',  # Changed from val_loss to val_accuracy
                patience=8,
                restore_best_weights=True,
                verbose=0,
                mode='max'
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.keras', save_best_only=True, monitor='val_accuracy', verbose=0, mode='max'
            )
        ],
        verbose=0
    )
    return history

# ============================================================================
# TEMPERATURE SCALING
# ============================================================================

def find_temperature(model, val_ds):
    section("Temperature Scaling Calibration")

    all_logits, all_labels = [], []
    for imgs, lbls in val_ds:
        probs = model(imgs, training=False).numpy()
        all_logits.append(np.log(np.clip(probs, 1e-7, 1.0)))
        all_labels.append(lbls.numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)

    best_T, best_nll = 1.0, float('inf')
    for T in np.arange(0.1, 5.05, 0.05):
        scaled = logits / T
        exp_s  = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs  = exp_s / exp_s.sum(axis=1, keepdims=True)
        nll    = -np.sum(labels * np.log(np.clip(probs, 1e-9, 1.0))) / len(labels)
        if nll < best_nll:
            best_nll, best_T = nll, T

    info(f"Best temperature : {best_T:.2f}")
    info(f"Val NLL          : {best_nll:.4f}")
    return float(best_T)

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, dataset, class_names, split_name, output_dir, temperature=1.0):
    section(f"Evaluation — {split_name}")

    all_preds, all_labels, all_confs = [], [], []
    for imgs, lbls in dataset:
        probs = model(imgs, training=False).numpy()
        if temperature != 1.0:
            logits = np.log(np.clip(probs, 1e-7, 1.0)) / temperature
            exp_s  = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs  = exp_s / exp_s.sum(axis=1, keepdims=True)
        all_preds.extend(np.argmax(probs, axis=1))
        all_labels.extend(np.argmax(lbls.numpy(), axis=1))
        all_confs.extend(np.max(probs, axis=1))

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    confs  = np.array(all_confs)

    acc  = (preds == labels).mean()
    mean = confs.mean()

    info(f"Accuracy         : {acc*100:.2f}%")
    info(f"Mean Confidence  : {mean*100:.2f}%")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix(labels, preds), display_labels=class_names).plot(
        ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f"Confusion Matrix — {split_name} (Acc: {acc*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cm_{split_name.lower()}.png"), dpi=150)
    plt.close()

    # Confidence Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(confs, bins=20, color='steelblue', alpha=0.8, edgecolor='white')
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean={mean:.1%}')
    plt.title(f"Confidence Distribution — {split_name}")
    plt.xlabel('Confidence'); plt.ylabel('Count')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confidence_{split_name.lower()}.png"), dpi=150)
    plt.close()

    return acc, mean

# ============================================================================
# PLOT HISTORY
# ============================================================================

def plot_history(h1, h2, output_dir):
    acc      = h1.history['accuracy']      + h2.history['accuracy']
    val_acc  = h1.history['val_accuracy']  + h2.history['val_accuracy']
    loss     = h1.history['loss']          + h2.history['loss']
    val_loss = h1.history['val_loss']      + h2.history['val_loss']
    split    = len(h1.history['accuracy'])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, train, val, title in zip(
        axes, [acc, loss], [val_acc, val_loss], ['Accuracy', 'Loss']
    ):
        ax.plot(train, 'b-', linewidth=2.5, label='Train')
        ax.plot(val, 'r-', linewidth=2.5, label='Validation')
        ax.axvline(split - 0.5, color='gray', linestyle=':', linewidth=2, label='Fine-tuning starts')
        ax.set_title(f'MobileNetV2 {title} — Cattle Disease (256×256)', fontsize=14)
        ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1) if title == 'Accuracy' else None

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MobileNetV2 Cattle Disease Trainer')
    parser.add_argument('--train-dir',  default=r"D:\project\new dataset\train")
    parser.add_argument('--val-dir',    default=r"D:\project\new dataset\valid")
    parser.add_argument('--test-dir',   default=r"D:\project\new dataset\test")
    parser.add_argument('--output-dir', default=r"D:\project\MobileNetV2_output")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load class names ──
    class_names = ['dermatophilosis', 'fmd', 'healthy', 'lumpy skin']

    print("\n" + "═"*70)
    print("  🐄 MobileNetV2 CATTLE DISEASE CLASSIFIER (Production Ready)")
    print("═"*70)
    info(f"Classes: {', '.join(class_names)}")
    info(f"Image size: {IMG_SIZE[0]}×{IMG_SIZE[1]} | Batch: {BATCH_SIZE}")

    # ── Datasets ──
    section("Loading Datasets")
    def load_ds(path, shuffle):
        return tf.keras.utils.image_dataset_from_directory(
            path, labels='inferred', label_mode='categorical',
            class_names=class_names, image_size=IMG_SIZE,
            batch_size=BATCH_SIZE, shuffle=shuffle
        )

    train_ds = load_ds(args.train_dir, True)
    val_ds   = load_ds(args.val_dir, False)
    test_ds  = load_ds(args.test_dir, False)
    ok("Datasets loaded ✓")

    augment = make_augmentation_layer()
    def pre_train(img, lbl): return tf.keras.applications.mobilenet_v2.preprocess_input(augment(tf.cast(img, tf.float32), training=True)), lbl
    def pre_eval(img, lbl):  return tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(img, tf.float32)), lbl

    train_ds = train_ds.map(pre_train, num_parallel_calls=4).cache().prefetch(buffer_size=2)
    val_ds   = val_ds.map(pre_eval, num_parallel_calls=4).cache().prefetch(buffer_size=2)
    test_ds  = test_ds.map(pre_eval, num_parallel_calls=4).cache().prefetch(buffer_size=2)

    # ── Class weights ──
    class_weight = get_class_weights(args.train_dir, class_names)

    # ── Training ──
    section("Building MobileNetV2")
    model, base_model = build_model(trainable_base=False)
    ok(f"Model ready: {model.count_params():,} params")

    h1 = train_phase(model, train_ds, val_ds, PHASE1_EPOCHS, PHASE1_LR, class_weight, "Phase 1: Head Training", is_phase2=False)
    
    total = len(base_model.layers)
    unfreeze = int(total * UNFREEZE_FROM_PCT)
    for layer in base_model.layers[unfreeze:]: layer.trainable = True
    
    h2 = train_phase(model, train_ds, val_ds, PHASE2_EPOCHS, PHASE2_LR, class_weight, f"Phase 2: Fine-tune {total-unfreeze}/{total} layers", is_phase2=True)

    # ── Temperature ──
    temperature = find_temperature(model, val_ds) if TEMPERATURE_CALIBRATION else 1.0

    # ── Save ──
    section("Saving Results")
    model.save(os.path.join(args.output_dir, 'best_mobilenetv2.keras'))
    ok("Model saved ✓")

    with open(os.path.join(args.output_dir, 'temperature.json'), 'w') as f:
        json.dump({'temperature': temperature}, f, indent=2)
    ok("Temperature saved ✓")

    plot_history(h1, h2, args.output_dir)

    # ── Evaluate ──
    val_acc, val_conf = evaluate(model, val_ds, class_names, "Validation", args.output_dir, temperature)
    test_acc, test_conf = evaluate(model, test_ds, class_names, "Test", args.output_dir, temperature)

    # ── FINAL SUMMARY ──
    print("\n" + "═"*70)
    print("  🏆 FINAL RESULTS")
    print("═"*70)
    print(f"  {'Metric':<20} | {'Validation':^12} | {'Test':^10}")
    print(f"  {'─'*20} | {'─'*12} | {'─'*10}")
    print(f"  {'Accuracy':<20} | {val_acc*100:>10.1f}% | {test_acc*100:>8.1f}%")
    print(f"  {'Confidence':<20} | {val_conf*100:>10.1f}% | {test_conf*100:>8.1f}%")
    print(f"  {'Temperature':<20} | {temperature:>10.2f} | {'✓':^10}")
    print("\n🎉 Training COMPLETE! Check output folder for all files.")
    print("═"*70)

if __name__ == '__main__':
    main()