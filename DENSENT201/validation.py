"""
DenseNet201 - Validation Script
Validate model on validation dataset with visualizations
"""

import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import argparse

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def validate_model(model_path, val_dir, output_dir, class_names_path):
    """Validate model with predictions."""
    os.makedirs(output_dir, exist_ok=True)

    # Load model and class names
    model = tf.keras.models.load_model(model_path)
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    print(f'✓ Model: {model_path}')
    print(f'✓ Classes: {class_names}')

    # Create validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.densenet.preprocess_input(image)
        return image, label

    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Evaluate
    print('\nEvaluating on validation set...')
    loss, acc, top2 = model.evaluate(val_ds, verbose=0)
    print(f'Val Accuracy: {acc:.4f}')
    print(f'Val Top-2 Accuracy: {top2:.4f}')
    print(f'Val Loss: {loss:.4f}')

    # Predictions
    print('\nGenerating predictions...')
    y_true = []
    y_pred = []
    confidences = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        confidences.extend(np.max(preds, axis=1))

    # Classification report
    print('\n' + '='*60)
    print('Classification Report')
    print('='*60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Validation', fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f'✓ Confusion matrix saved')

    # Confidence distribution
    plt.figure(figsize=(10, 5))
    plt.hist(confidences, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Confidence Distribution', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_dist.png'), dpi=150)
    plt.close()
    print(f'✓ Confidence distribution saved')

    # Per-class accuracy
    per_class_acc = []
    print('\nPer-class Accuracy:')
    for i in range(len(class_names)):
        mask = np.array(y_true) == i
        if mask.sum() > 0:
            class_acc = (np.array(y_pred)[mask] == i).mean()
            per_class_acc.append(class_acc)
            print(f'  {class_names[i]}: {class_acc:.4f}')

    # Save validation metrics
    metrics = {
        'accuracy': float(acc),
        'top2_accuracy': float(top2),
        'loss': float(loss),
        'per_class_accuracy': {cn: float(ca) for cn, ca in zip(class_names, per_class_acc)}
    }
    with open(os.path.join(output_dir, 'validation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\n✓ Results saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=r'D:\project\models\best_densenet201.keras',
                        help='Path to model')
    parser.add_argument('--val-dir', type=str, default=r'D:\project\new dataset\valid',
                        help='Path to validation directory')
    parser.add_argument('--output-dir', type=str, default=r'D:\project\densenet201\results',
                        help='Output directory')
    parser.add_argument('--class-names', type=str, default=r'D:\project\models\class_names.json',
                        help='Path to class names json')

    args = parser.parse_args()

    validate_model(args.model, args.val_dir, args.output_dir, args.class_names)
