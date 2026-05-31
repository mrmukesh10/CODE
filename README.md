# MobileNetV2 Livestock Disease Detection Pipeline - Python 3.10

Automated disease detection model for cattle using deep learning.

**Detectable Classes (4):**
- 🦠 **Dermatophilosis** - Bacterial skin infection
- 🦠 **FMD** (Foot and Mouth Disease) - Viral disease affecting mouth & hooves
- 🦠 **Lumpy Skin** - Viral skin disease
- ✅ **Healthy** - Normal cattle (control class)

Complete training, testing, and validation scripts for MobileNetV2 model training, compatible with Python 3.10.

## Overview

This pipeline consists of three independent Python scripts:

- **[train.py](train.py)** - Train the MobileNetV2 model (Phase 1: head only, Phase 2: fine-tuning)
- **[test.py](test.py)** - Evaluate the trained model on test set with confusion matrix
- **[validation.py](validation.py)** - Validate model on validation set with sample visualizations

## Features

✓ Two-phase training strategy (head training + fine-tuning)
✓ Cosine learning rate decay with warm restarts
✓ Class balancing and weighted loss
✓ MixUp data augmentation
✓ Automatic best model checkpointing
✓ Confusion matrix and classification reports
✓ Sample prediction visualization
✓ Python 3.10 compatible
✓ TensorFlow/Keras based

## Requirements

```bash
pip install tensorflow scikit-learn matplotlib seaborn opencv-python numpy
```

Or use the compatible versions:
```bash
pip install tensorflow==2.15.0 scikit-learn==1.3.2 matplotlib==3.8.2 seaborn==0.13.0 opencv-python==4.8.1 numpy==1.24.3
```

## Dataset Structure

Your dataset should be organized as follows:

```
new dataset/
├── train/                    (Training: ~60-70% of data)
│   ├── dermatophilosis/     
│   ├── fmd/                 
│   ├── healthy/             
│   └── lumpy skin/          
├── valid/                    (Validation: ~15-20% of data)
│   ├── dermatophilosis/
│   ├── fmd/
│   ├── healthy/
│   └── lumpy skin/
└── test/                     (Testing: ~15-20% of data)
    ├── dermatophilosis/
    ├── fmd/
    ├── healthy/
    └── lumpy skin/
```

**Requirements:**
- Minimum 10-20 images per class per split
- Recommended 50+ images per class per split
- Supported formats: `.jpg`, `.jpeg`, `.png`
- Image size: Any size (will be resized to 256×256)

## Usage

### 1. Training

Train the model on cattle disease dataset:

```bash
python train.py
```

Or with custom paths:

```bash
python train.py --train-dir "new dataset/train" --val-dir "new dataset/valid" --output-dir "./models"
```

**Arguments:**
- `--train-dir`: Path to training directory (default: `new dataset/train`)
- `--val-dir`: Path to validation directory (default: `new dataset/valid`)
- `--output-dir`: Output directory for model (default: `./models`)

**Output:**
- `best_mobilenetv2.keras` - Best trained model
- `class_names.json` - Class names: dermatophilosis, fmd, healthy, lumpy skin
- `training_curves.png` - Accuracy and loss plots across both phases

**Runtime:**
- GPU (T4): ~25-35 minutes
- CPU: ~1.5-2 hours

### 2. Testing

Evaluate on test set with detailed metrics:

```bash
python test.py
```

Or with custom paths:

```bash
python test.py --model "./models/best_mobilenetv2.keras" --test-dir "new dataset/test" --output-dir "./results"
```

**Arguments:**
- `--model`: Path to trained model (default: `./models/best_mobilenetv2.keras`)
- `--test-dir`: Path to test directory (default: `new dataset/test`)
- `--class-names`: Path to class names file (default: `./models/class_names.json`)
- `--output-dir`: Output directory (default: `./results`)

**Output:**
- `confusion_matrix.png` - Heatmap showing per-disease accuracy
- `classification_report.txt` - Precision, Recall, F1-Score for each disease

**Metrics Generated:**
- Overall Accuracy
- Top-2 Accuracy (% times correct answer in top 2 predictions)
- Per-disease accuracy (Dermatophilosis, FMD, Healthy, Lumpy Skin)

### 3. Validation

Analyze model behavior on validation set:

```bash
python validation.py
```

Or with custom paths:

```bash
python validation.py --model "./models/best_mobilenetv2.keras" --val-dir "new dataset/valid" --output-dir "./validation_results"
```

**Arguments:**
- `--model`: Path to trained model (default: `./models/best_mobilenetv2.keras`)
- `--val-dir`: Path to validation directory (default: `new dataset/valid`)
- `--class-names`: Path to class names file (default: `./models/class_names.json`)
- `--output-dir`: Output directory (default: `./validation_results`)

**Output:**
- `sample_predictions.png` - Visual predictions with confidence scores (green=correct, red=wrong)
- `validation_report.txt` - Validation statistics and class distribution

## Configuration (train.py)

Edit these settings in `train.py`:

```python
IMG_SIZE = (256, 256)          # Input image size
BATCH_SIZE = 32                # Batch size (reduce to 16 if OOM)
NUM_CLASSES = 4                # Number of classes

# Phase 1 (head training)
PHASE1_EPOCHS = 15             # Number of epochs for phase 1
PHASE1_LR = 1e-3               # Learning rate for phase 1

# Phase 2 (fine-tuning)
PHASE2_EPOCHS = 35             # Number of epochs for phase 2
PHASE2_LR = 1e-5               # Learning rate for phase 2
UNFREEZE_FROM_PCT = 0.70       # Unfreeze last 30% of base

# Regularization
DROPOUT_RATE = 0.4             # Dropout rate
L2_REG = 1e-4                  # L2 regularization
LABEL_SMOOTHING = 0.1          # Label smoothing
USE_MIXUP = True               # Use MixUp augmentation
MIXUP_ALPHA = 0.2              # MixUp alpha parameter
```

## Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Head**: 
  - Global Average Pooling
  - Dense(256, ReLU) + BatchNorm + Dropout(0.4)
  - Dense(128, ReLU) + Dropout(0.2)
  - Dense(4, Softmax)

## Training Overview

### Phase 1: Head Training (Frozen Base)
- Trains only the custom head (top layers)
- Base MobileNetV2 model remains frozen
- Uses cosine decay learning rate schedule
- Early stopping patience: 7 epochs

### Phase 2: Fine-tuning
- Unfreezes last 30% of base model layers
- BatchNormalization layers remain frozen
- Uses cosine decay with warm restarts
- Early stopping patience: 12 epochs
- ReduceLROnPlateau callback for adaptive learning

## Performance Tips

1. **Batch Size**: If you get OOM errors, reduce `BATCH_SIZE` from 32 to 16
2. **GPU**: Training uses GPU if available. Check with `nvidia-smi`
3. **Data Augmentation**: MixUp is enabled by default for better generalization
4. **Class Imbalance**: Automatic class weight calculation handles imbalanced datasets
5. **Learning Rate**: Cosine decay provides smooth learning rate transitions

## Output Files

### After Training:
- `models/best_mobilenetv2.keras` - Trained model
- `models/class_names.json` - Class mapping

### After Testing:
- `results/confusion_matrix.png` - Confusion matrix
- `results/classification_report.txt` - Detailed metrics

### After Validation:
- `validation_results/sample_predictions.png` - Sample predictions
- `validation_results/validation_report.txt` - Validation metrics

## Python Version

All scripts are compatible with:
- Python 3.10+ ✓
- Python 3.11 ✓
- Python 3.12 ✓

## Troubleshooting

### "CUDA not available"
- Check NVIDIA GPU drivers: `nvidia-smi`
- Install CUDA-compatible TensorFlow: `pip install tensorflow-gpu`

### "Out of Memory (OOM)"
- Reduce BATCH_SIZE in train.py from 32 to 16
- Clear GPU memory: Restart Python/notebook kernel

### "Model not found"
- Ensure training has completed successfully
- Verify model path is correct
- Check that `best_mobilenetv2.keras` exists in output directory

### "No images found"
- Verify dataset directory structure
- Check file extensions are `.jpg`, `.jpeg`, or `.png`
- Ensure at least one image exists in each class folder

## Next Steps

1. Prepare your dataset in the required structure
2. Run `train.py` to train the model
3. Run `test.py` to evaluate on test set
4. Run `validation.py` to analyze validation performance
5. Review confusion matrices and reports to identify weak classes
6. Adjust hyperparameters if needed and retrain

## References

- [TensorFlow Keras](https://www.tensorflow.org/api_docs/python/keras)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Class Weighting in Keras](https://www.tensorflow.org/api_docs/python/tf/keras/models/Model#fit)

## License

This code is provided as-is for educational and research purposes.
