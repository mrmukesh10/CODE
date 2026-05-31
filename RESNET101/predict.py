"""
ResNet101 - Single Image Prediction
Predict disease class for a single cattle image
"""

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)


def predict_single_image(model_path, image_path, class_names_path, output_dir=None):
    """Predict disease for a single image."""
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    print(f'✓ Model loaded: {model_path}')

    # Load class names
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    print(f'✓ Classes: {class_names}')

    # Load and preprocess image
    print(f'\nLoading image: {image_path}')
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img.resize(IMG_SIZE))
    
    # Preprocess
    img_batch = np.expand_dims(img_array, axis=0).astype(np.float32)
    img_batch = tf.keras.applications.resnet.preprocess_input(img_batch)

    # Predict
    print('Making prediction...')
    predictions = model.predict(img_batch, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]

    # Display results
    print('\n' + '='*60)
    print('PREDICTION RESULTS - ResNet101')
    print('='*60)
    print(f'Predicted Class: {class_names[predicted_class].upper()}')
    print(f'Confidence: {confidence*100:.2f}%')
    print(f'\nAll Predictions:')
    for i, (class_name, prob) in enumerate(zip(class_names, predictions)):
        print(f'  {class_name:<20} {prob*100:>6.2f}%')
    print('='*60 + '\n')

    # Save visualization if output dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Image with prediction
        ax1.imshow(img)
        ax1.set_title(f'Predicted: {class_names[predicted_class].upper()}\nConfidence: {confidence*100:.2f}%', 
                     fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Confidence bars
        colors = ['red' if i != predicted_class else 'green' for i in range(len(class_names))]
        ax2.barh(class_names, predictions, color=colors, alpha=0.7)
        ax2.set_xlabel('Confidence Score')
        ax2.set_title('Prediction Confidence by Class')
        ax2.set_xlim(0, 1)
        
        for i, v in enumerate(predictions):
            ax2.text(v + 0.02, i, f'{v*100:.1f}%', va='center')
        
        plt.tight_layout()
        
        # Save image
        output_path = os.path.join(output_dir, 'prediction_result.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✓ Visualization saved: {output_path}')
        
        # Save JSON result
        result = {
            'image': image_path,
            'predicted_class': class_names[predicted_class],
            'confidence': float(confidence),
            'all_predictions': {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
        }
        json_path = os.path.join(output_dir, 'prediction_result.json')
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'✓ Result saved: {json_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict disease for a single image')
    parser.add_argument('image', nargs='?', help='Path to the input image')
    parser.add_argument('--model', default=r'D:\project\models\best_resnet101.keras', help='Model path')
    parser.add_argument('--class-names', default=r'D:\project\models\class_names.json', help='Class names JSON')
    parser.add_argument('--output-dir', default=r'D:\project\resnet101\results', help='Output directory for results')
    args = parser.parse_args()

    print('\n' + '='*60)
    print('  ResNet101 - SINGLE IMAGE PREDICTION')
    print('='*60)
    
    if not args.image:
        print('\nUsage: python predict.py <image_path>')
        print('Example: python predict.py "C:\\path\\to\\image.jpg"')
        exit(1)
    
    if not os.path.exists(args.image):
        print(f'✗ Image not found: {args.image}')
        exit(1)
    
    predict_single_image(args.model, args.image, args.class_names, args.output_dir)
