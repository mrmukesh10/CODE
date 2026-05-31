"""
ResNet101 - Validation Script
Evaluate model performance on validation dataset during training
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


def validate_model(model_path, val_dir, output_dir, class_names_path):
    """Validate model on validation set."""
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
        image = tf.keras.applications.resnet.preprocess_input(image)
        return image, label

    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.cache()
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Print validation set summary
    print('\nValidation Set Summary:')
    per_class = {}
    for c in class_names:
        count = len([f for f in os.listdir(os.path.join(val_dir, c)) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        per_class[c] = count
        print(f'  {c:<20}: {count:4d} images')
    total = sum(per_class.values())
    print(f'  {"TOTAL":<20}: {total:4d} images')

    # Evaluate
    print('\n' + '='*60)
    print('Evaluating on Validation Set:')
    print('='*60)
    val_loss, val_acc, val_top2 = model.evaluate(val_ds, verbose=1)

    print(f'\n✓ Validation Results:')
    print(f'  Accuracy      : {val_acc * 100:.2f}%')
    print(f'  Top-2 Accuracy: {val_top2 * 100:.2f}%')
    print(f'  Loss          : {val_loss:.4f}')

    # Collect predictions
    print('\nGenerating predictions...')
    y_true_list, y_pred_list = [], []

    for images, labels in val_ds:
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
    plt.title('Validation Confusion Matrix - ResNet101')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'validation_confusion_matrix.png')
    plt.savefig(cm_path, dpi=100)
    print(f'✓ Confusion matrix saved: {cm_path}')
    plt.close()

    # Classification report
    report = classification_report(y_true_classes, y_pred_classes,
                                   target_names=class_names, output_dict=True)

    print('\nDetailed Classification Report:')
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

    # Per-class accuracy
    print('\nPer-Class Accuracy:')
    for i, class_name in enumerate(class_names):
        class_correct = np.sum((y_true_classes == i) & (y_pred_classes == i))
        class_total = np.sum(y_true_classes == i)
        class_acc = class_correct / class_total * 100 if class_total > 0 else 0
        print(f'  {class_name:<20}: {class_acc:6.2f}% ({class_correct}/{class_total})')

    # Save results
    results = {
        'accuracy': float(val_acc),
        'top_2_accuracy': float(val_top2),
        'loss': float(val_loss),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'class_distribution': per_class
    }

    results_path = os.path.join(output_dir, 'validation_metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✓ Results saved: {results_path}')

    return results


def main():
    """Main function."""
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(description='Validate ResNet101 Model')
    parser.add_argument(
        '--model-path',
        type=str,
        default=os.path.join(script_dir, 'best_resnet101.keras'),
        help='Path to trained model'
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
        default=os.path.join(script_dir, 'results', 'validation'),
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
    print('ResNet101 - Model Validation')
    print('='*60 + '\n')

    validate_model(args.model_path, args.val_dir, args.output_dir, args.class_names_path)

    print('\n✓ Validation complete!')


if __name__ == '__main__':
    main()
