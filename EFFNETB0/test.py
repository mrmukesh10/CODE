"""
EfficientNetB0 - Testing Script
Evaluate model performance on test dataset
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


def evaluate_model(model_path, test_dir, output_dir, class_names_path):
    """Evaluate model on test set."""
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = tf.keras.models.load_model(model_path)
    print(f'[OK] Model loaded: {model_path}')

    # Load class names
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    print(f'[OK] Classes: {class_names}')

    # Create test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image, label

    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Evaluate
    print('\nEvaluating on test set...')
    loss, acc, top2 = model.evaluate(test_ds, verbose=0)
    print(f'Test Accuracy: {acc:.4f}')
    print(f'Test Top-2 Accuracy: {top2:.4f}')
    print(f'Test Loss: {loss:.4f}')

    # Predictions
    print('\nGenerating predictions...')
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

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
    plt.title('Confusion Matrix - EfficientNetB0', fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print('[OK] Confusion matrix saved')

    # Save metrics
    metrics = {
        'accuracy': float(acc),
        'top2_accuracy': float(top2),
        'loss': float(loss)
    }
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\n✓ Results saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=r'D:\project\models\best_efficientnetb0.keras',
                        help='Path to model')
    parser.add_argument('--test-dir', type=str, default=r'D:\project\new dataset\test',
                        help='Path to test directory')
    parser.add_argument('--output-dir', type=str, default=r'D:\project\efficientnetb0\results',
                        help='Output directory')
    parser.add_argument('--class-names', type=str, default=r'D:\project\models\class_names.json',
                        help='Path to class names json')

    args = parser.parse_args()

    evaluate_model(args.model, args.test_dir, args.output_dir, args.class_names)
