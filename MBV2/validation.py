"""
MobileNetV2 - Validation Script
Evaluate model performance on validation dataset
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import argparse

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def evaluate_validation(model_path, val_dir, output_dir, class_names_path):
    """Evaluate model on validation set."""
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = tf.keras.models.load_model(model_path)
    print(f'✓ Model loaded: {model_path}')

    # Load class names
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
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
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Evaluate
    print('\nEvaluating on validation set...')
    loss, acc, top2 = model.evaluate(val_ds, verbose=0)
    print(f'Validation Accuracy: {acc:.4f}')
    print(f'Validation Top-2 Accuracy: {top2:.4f}')
    print(f'Validation Loss: {loss:.4f}')

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
    plt.title('Confusion Matrix - MobileNetV2 Validation', fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_confusion_matrix.png'), dpi=150)
    plt.close()
    print(f'✓ Confusion matrix saved')

    # Confidence histogram
    plt.figure(figsize=(10, 5))
    plt.hist(confidences, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution - Validation')
    plt.axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=150)
    plt.close()
    print(f'✓ Confidence histogram saved')

    # Save metrics
    metrics = {
        'accuracy': float(acc),
        'top2_accuracy': float(top2),
        'loss': float(loss),
        'mean_confidence': float(np.mean(confidences)),
        'std_confidence': float(np.std(confidences))
    }
    with open(os.path.join(output_dir, 'validation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\n✓ Results saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate MobileNetV2')
    parser.add_argument('--model', default=r'D:\project\models\best_mobilenetv2.keras', help='Model path')
    parser.add_argument('--val-dir', default=r'D:\project\new dataset\valid', help='Validation directory')
    parser.add_argument('--output-dir', default=r'D:\project\mobilenetv2\results', help='Output directory')
    parser.add_argument('--class-names', default=r'D:\project\models\class_names.json', help='Class names JSON')
    args = parser.parse_args()

    print('\n' + '='*70)
    print('  MobileNetV2 - VALIDATION EVALUATION')
    print('='*70 + '\n')
    
    evaluate_validation(args.model, args.val_dir, args.output_dir, args.class_names)
