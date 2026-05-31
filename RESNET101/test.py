"""
ResNet101 - Testing Script
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
    print(f'✓ Model loaded: {model_path}')

    # Load class names
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    print(f'✓ Classes: {class_names}')

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
        image = tf.keras.applications.resnet.preprocess_input(image)
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
    y_true_list, y_pred_list = [], []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_list.append(labels.numpy())
        y_pred_list.append(preds)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    # Get class predictions
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - ResNet101')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=100)
    print(f'✓ Confusion matrix saved: {cm_path}')
    plt.close()

    # Classification report
    report = classification_report(y_true_classes, y_pred_classes, 
                                   target_names=class_names, output_dict=True)
    
    print('\nClassification Report:')
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

    # Save results
    results = {
        'accuracy': float(acc),
        'top_2_accuracy': float(top2),
        'loss': float(loss),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

    results_path = os.path.join(output_dir, 'test_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'✓ Results saved: {results_path}')

    return results


def main():
    """Main function."""
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(description='Test ResNet101 Model')
    parser.add_argument(
        '--model-path',
        type=str,
        default=os.path.join(script_dir, 'best_resnet101.keras'),
        help='Path to trained model'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default=os.path.join(project_dir, 'new dataset', 'test'),
        help='Test directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.path.join(script_dir, 'results'),
        help='Output directory'
    )
    parser.add_argument(
        '--class-names-path',
        type=str,
        default=os.path.join(project_dir, 'models', 'class_names.json'),
        help='Path to class names JSON'
    )

    args = parser.parse_args()

    print('\n' + '='*60)
    print('ResNet101 - Model Testing')
    print('='*60 + '\n')

    evaluate_model(args.model_path, args.test_dir, args.output_dir, args.class_names_path)

    print('\n✓ Testing complete!')


if __name__ == '__main__':
    main()
