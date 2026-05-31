"""
Ensemble Prediction - All 4 Models
Combines predictions from DenseNet201, EfficientNetB0, ResNet101, and MobileNetV2
Provides overall disease vs healthy classification with confidence scores
"""

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Image size for each model
IMG_SIZES = {
    'densenet201': (224, 224),
    'efficientnetb0': (224, 224),
    'resnet101': (224, 224),
    'mobilenetv2': (224, 224)
}

CLASS_NAMES = ['dermatophilosis', 'fmd', 'healthy', 'lumpy skin']
DISEASE_CLASSES = ['dermatophilosis', 'fmd', 'lumpy skin']
HEALTHY_CLASS = 'healthy'


def load_models(models_dir):
    """Load all 4 pre-trained models."""
    models = {}
    model_names = ['best_densenet201', 'best_efficientnetb0', 'best_resnet101', 'best_mobilenetv2']
    
    print('\n' + '='*70)
    print('  Loading Models')
    print('='*70)
    
    for model_name in model_names:
        model_path = os.path.join(models_dir, f'{model_name}.keras')
        if os.path.exists(model_path):
            models[model_name.replace('best_', '').replace('.keras', '')] = tf.keras.models.load_model(model_path)
            print(f'✓ {model_name} loaded')
        else:
            print(f'✗ {model_name} NOT FOUND: {model_path}')
    
    return models


def preprocess_image(image_path, model_name):
    """Load and preprocess image for specific model."""
    img = Image.open(image_path).convert('RGB')
    img_size = IMG_SIZES[model_name]
    img_array = np.array(img.resize(img_size)).astype(np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Apply model-specific preprocessing
    if model_name == 'densenet201':
        img_batch = tf.keras.applications.densenet.preprocess_input(img_batch)
    elif model_name == 'efficientnetb0':
        img_batch = tf.keras.applications.efficientnet.preprocess_input(img_batch)
    elif model_name == 'resnet101':
        img_batch = tf.keras.applications.resnet.preprocess_input(img_batch)
    elif model_name == 'mobilenetv2':
        img_batch = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
    
    return img_batch


def predict_ensemble(image_path, models, output_dir=None):
    """Generate predictions from all models and ensemble them."""
    
    print('\n' + '='*70)
    print('  ENSEMBLE PREDICTION - ALL 4 MODELS')
    print('='*70)
    print(f'\nImage: {image_path}')
    
    if not os.path.exists(image_path):
        print(f'✗ Image not found: {image_path}')
        return
    
    # Load original image for display
    original_img = Image.open(image_path).convert('RGB')
    
    # Get predictions from all models
    all_predictions = {}
    all_confidences = {}
    
    print('\nModel Predictions:')
    print('-' * 70)
    
    for model_name, model in models.items():
        # Preprocess image for this model
        img_batch = preprocess_image(image_path, model_name)
        
        # Get predictions
        predictions = model.predict(img_batch, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = predictions[predicted_class_idx]
        
        all_predictions[model_name] = predictions
        all_confidences[model_name] = {
            'class': predicted_class,
            'confidence': float(confidence)
        }
        
        print(f'\n{model_name.upper()}:')
        print(f'  Predicted: {predicted_class.upper()}')
        print(f'  Confidence: {confidence*100:.2f}%')
        for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, predictions)):
            print(f'    {class_name:<20} {prob*100:>6.2f}%')
    
    # Ensemble: Average predictions
    ensemble_predictions = np.mean([all_predictions[name] for name in models.keys()], axis=0)
    ensemble_class_idx = np.argmax(ensemble_predictions)
    ensemble_class = CLASS_NAMES[ensemble_class_idx]
    ensemble_confidence = ensemble_predictions[ensemble_class_idx]
    
    # Disease vs Healthy classification
    disease_prob = np.sum(ensemble_predictions[[i for i, c in enumerate(CLASS_NAMES) if c in DISEASE_CLASSES]])
    healthy_prob = ensemble_predictions[CLASS_NAMES.index(HEALTHY_CLASS)]
    
    if disease_prob > healthy_prob:
        overall_class = 'DISEASE'
        overall_confidence = disease_prob
    else:
        overall_class = 'HEALTHY'
        overall_confidence = healthy_prob
    
    # Display ensemble results
    print('\n' + '='*70)
    print('  ENSEMBLE RESULTS (Average of All 4 Models)')
    print('='*70)
    print(f'\nClass Predictions (Averaged):')
    for class_name, prob in zip(CLASS_NAMES, ensemble_predictions):
        print(f'  {class_name:<20} {prob*100:>6.2f}%')
    
    print(f'\n{"─"*70}')
    print(f'  Primary Prediction:  {ensemble_class.upper()}')
    print(f'  Confidence:          {ensemble_confidence*100:.2f}%')
    print(f'{"─"*70}')
    
    print(f'\n  OVERALL CLASSIFICATION:')
    print(f'  Status:              {overall_class}')
    print(f'  Confidence:          {overall_confidence*100:.2f}%')
    print(f'  Disease Probability: {disease_prob*100:.2f}%')
    print(f'  Healthy Probability: {healthy_prob*100:.2f}%')
    print('='*70 + '\n')
    
    # Save visualization if output dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(18, 10))
        
        # Original image
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(original_img)
        ax1.set_title('Input Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Individual model predictions (top 3)
        for idx, (model_name, model) in enumerate(list(models.items())[:3]):
            ax = plt.subplot(2, 3, idx+2)
            preds = all_predictions[model_name]
            colors = ['green' if i == np.argmax(preds) else 'skyblue' for i in range(len(CLASS_NAMES))]
            ax.barh(CLASS_NAMES, preds, color=colors, alpha=0.7)
            ax.set_xlabel('Confidence')
            ax.set_title(f'{model_name.upper()}\n({all_confidences[model_name]["class"].upper()})', 
                        fontsize=10, fontweight='bold')
            ax.set_xlim(0, 1)
            for i, v in enumerate(preds):
                ax.text(v + 0.02, i, f'{v*100:.1f}%', va='center', fontsize=9)
        
        # Ensemble predictions
        ax5 = plt.subplot(2, 3, 5)
        colors = ['green' if i == ensemble_class_idx else 'lightblue' for i in range(len(CLASS_NAMES))]
        ax5.barh(CLASS_NAMES, ensemble_predictions, color=colors, alpha=0.8)
        ax5.set_xlabel('Confidence')
        ax5.set_title(f'ENSEMBLE AVERAGE\n({ensemble_class.upper()}: {ensemble_confidence*100:.1f}%)', 
                     fontsize=11, fontweight='bold', color='darkgreen')
        ax5.set_xlim(0, 1)
        for i, v in enumerate(ensemble_predictions):
            ax5.text(v + 0.02, i, f'{v*100:.1f}%', va='center', fontsize=9, fontweight='bold')
        
        # Overall disease vs healthy
        ax6 = plt.subplot(2, 3, 6)
        categories = ['DISEASE', 'HEALTHY']
        probs = [disease_prob, healthy_prob]
        colors_pie = ['#ff6b6b' if overall_class == 'DISEASE' else 'lightgray',
                     '#51cf66' if overall_class == 'HEALTHY' else 'lightgray']
        wedges, texts, autotexts = ax6.pie(probs, labels=categories, autopct='%1.1f%%',
                                            colors=colors_pie, startangle=90, textprops={'fontsize': 11})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax6.set_title(f'OVERALL: {overall_class}\n(Confidence: {overall_confidence*100:.1f}%)', 
                     fontsize=11, fontweight='bold',
                     color='darkred' if overall_class == 'DISEASE' else 'darkgreen')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(output_dir, 'ensemble_prediction.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✓ Visualization saved: {viz_path}')
        
        # Save detailed JSON results
        result = {
            'image': image_path,
            'ensemble_prediction': {
                'primary_class': ensemble_class,
                'confidence': float(ensemble_confidence),
                'all_classes': {CLASS_NAMES[i]: float(ensemble_predictions[i]) for i in range(len(CLASS_NAMES))}
            },
            'overall_classification': {
                'status': overall_class,
                'confidence': float(overall_confidence),
                'disease_probability': float(disease_prob),
                'healthy_probability': float(healthy_prob)
            },
            'individual_model_predictions': {
                model_name: {
                    'predicted_class': all_confidences[model_name]['class'],
                    'confidence': all_confidences[model_name]['confidence'],
                    'all_predictions': {CLASS_NAMES[i]: float(all_predictions[model_name][i]) 
                                       for i in range(len(CLASS_NAMES))}
                }
                for model_name in models.keys()
            }
        }
        
        json_path = os.path.join(output_dir, 'ensemble_prediction.json')
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'✓ Detailed results saved: {json_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble prediction using all 4 models')
    parser.add_argument('image', nargs='?', help='Path to input image')
    parser.add_argument('--models-dir', default=r'D:\project\models', help='Models directory')
    parser.add_argument('--output-dir', default=r'D:\project\mobilenetv2\results', 
                       help='Output directory for results')
    args = parser.parse_args()
    
    # If no image provided, show usage
    if not args.image:
        print('\n' + '='*70)
        print('  ENSEMBLE PREDICTION - All 4 Models')
        print('='*70)
        print('\nUsage:')
        print('  python ensemble_predict.py <image_path>')
        print('\nExample:')
        print('  python ensemble_predict.py "C:\\Users\\LENOVO\\Downloads\\der4.jpg"')
        print('  python ensemble_predict.py "D:\\project\\new dataset\\test\\healthy\\image.jpg"')
        print('\nOptions:')
        print('  --models-dir      Directory containing model files (default: D:\\project\\models)')
        print('  --output-dir      Directory to save results (default: D:\\project\\mobilenetv2\\results)')
        print('='*70 + '\n')
        exit(1)
    
    # Load models
    models = load_models(args.models_dir)
    
    if len(models) < 4:
        print(f'\n✗ Warning: Only {len(models)} models found. Expected 4.')
    
    # Generate ensemble prediction
    predict_ensemble(args.image, models, args.output_dir)
