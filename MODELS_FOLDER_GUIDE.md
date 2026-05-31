# Model Folder Structure & Saved Files

After running the complete training pipeline, your `models` folder will contain:

## 📁 Complete Folder Structure

```
d:\project\
├── models/                              ← Main model folder (created automatically)
│   ├── best_mobilenetv2.keras          ← Trained model (for inference)
│   ├── class_names.json                ← Class mapping (JSON)
│   ├── training_metrics.txt            ← Phase 1 & Phase 2 training metrics
│   ├── test_metrics.txt                ← Test set evaluation metrics
│   ├── validation_metrics.txt          ← Validation set metrics
│   └── training_curves.png             ← Accuracy & Loss graphs
│
├── results/                             ← Test results (optional)
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
└── validation_results/                  ← Validation results (optional)
    ├── sample_predictions.png
    └── validation_report.txt
```

---

## 📊 File Descriptions

### 1. **best_mobilenetv2.keras**
- Trained model file ready for inference
- Use this to make predictions on new images
- File size: ~40-50 MB

### 2. **class_names.json**
Example content:
```json
[
  "dermatophilosis",
  "fmd",
  "healthy",
  "lumpy skin"
]
```

### 3. **training_metrics.txt**
Contains:
- Dataset paths used
- Configuration (image size, batch size, epochs)
- Phase 1 training stats (accuracy, loss)
- Phase 2 fine-tuning stats (accuracy, loss)
- Class weights used
- Model parameters count

Example:
```
============================================================
MobileNetV2 TRAINING METRICS
============================================================

Dataset: new dataset/train
Training Date: 2026-04-09 14:30:45

CLASSES DETECTED:
  1. dermatophilosis
  2. fmd
  3. healthy
  4. lumpy skin

MODEL CONFIGURATION:
  Image Size: (256, 256)
  Batch Size: 32
  Total Params: 3,537,984
  Trainable Params: 268,804

PHASE 1 TRAINING (Head Only - Frozen Base):
  Epochs: 15
  Learning Rate: 0.001
  Best Validation Accuracy: 0.8234
  Best Training Accuracy: 0.8912
  Final Loss: 0.3421

PHASE 2 TRAINING (Fine-tuning - Unfrozen 30%):
  Epochs: 35
  Learning Rate: 0.00001
  Best Validation Accuracy: 0.8987
  Best Training Accuracy: 0.9456
  Final Loss: 0.2134
```

### 4. **test_metrics.txt**
Contains:
- Overall test accuracy
- Top-2 accuracy
- Test loss
- Per-class accuracy for each disease
- Confusion matrix (both counts and percentages)
- Classification metrics (Precision, Recall, F1-Score)

Example:
```
============================================================
MobileNetV2 TEST EVALUATION METRICS
============================================================

OVERALL TEST RESULTS:
  Overall Accuracy       : 89.34%
  Top-2 Accuracy         : 95.23%
  Test Loss              : 0.3421
  Total Test Images      : 200

PER-CLASS ACCURACY:
  dermatophilosis      : 88.50%  [GOOD]
  fmd                  : 91.22%  [GOOD]
  healthy              : 95.45%  [GOOD]
  lumpy skin           : 84.23%  [LOW]

CONFUSION MATRIX (Counts):
                dermatophilosis       fmd     healthy  lumpy skin
dermatophilosis:                 47         2         1          0
           fmd:                  3        58         1          1
       healthy:                  1         2        63          1
    lumpy skin:                  2         4         1         16
```

### 5. **validation_metrics.txt**
Contains:
- Validation set summary
- Overall validation accuracy
- Top-2 validation accuracy
- Validation loss
- Class distribution in validation set

Example:
```
============================================================
MobileNetV2 VALIDATION METRICS
============================================================

VALIDATION SET SUMMARY:
  dermatophilosis      :   15 images (25.0%)
  fmd                  :   15 images (25.0%)
  healthy              :   15 images (25.0%)
  lumpy skin           :   15 images (25.0%)
  TOTAL                :   60 images

VALIDATION RESULTS:
  Overall Accuracy : 89.87%
  Top-2 Accuracy   : 96.34%
  Validation Loss  : 0.2845
  Total Images     : 60
```

### 6. **training_curves.png**
Graph showing:
- Training accuracy vs Validation accuracy
- Training loss vs Validation loss
- Phase transition line (Phase 1 → Phase 2)

---

## 🚀 How to Use the Saved Files

### Make Predictions on New Image
```python
import tensorflow as tf
import json

# Load model
model = tf.keras.models.load_model('./models/best_mobilenetv2.keras')

# Load class names
with open('./models/class_names.json', 'r') as f:
    class_names = json.load(f)

# Load and preprocess image
img = tf.keras.preprocessing.image.load_img('test_cow.jpg', target_size=(256, 256))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
img_batch = tf.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_batch)
predicted_class = class_names[tf.argmax(predictions[0])]
confidence = tf.reduce_max(predictions[0]) * 100

print(f"Disease: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
```

### Review Metrics Programmatically
```python
# Read training metrics
with open('./models/training_metrics.txt', 'r') as f:
    training_info = f.read()
    print(training_info)

# Read test metrics
with open('./models/test_metrics.txt', 'r') as f:
    test_info = f.read()
    print(test_info)

# Read validation metrics
with open('./models/validation_metrics.txt', 'r') as f:
    val_info = f.read()
    print(val_info)
```

---

## 📈 Quick Reference - Expected Accuracy Values

For well-balanced dataset (50+ images per class per split):

| Metric | Phase 1 | Phase 2 | Test Set |
|--------|---------|---------|----------|
| **Training Accuracy** | 85-90% | 92-96% | N/A |
| **Validation Accuracy** | 75-82% | 85-92% | - |
| **Test Accuracy** | - | - | 85-95% |
| **Per-Class Range** | - | - | 80-95% |

---

## 📝 File Summary

| File | Format | Size | Purpose |
|------|--------|------|---------|
| best_mobilenetv2.keras | Keras Model | 40-50 MB | Model inference |
| class_names.json | JSON | <1 KB | Class mapping |
| training_metrics.txt | Text | 2-5 KB | Training stats |
| test_metrics.txt | Text | 3-8 KB | Test evaluation |
| validation_metrics.txt | Text | 1-3 KB | Validation stats |
| training_curves.png | Image | 100-200 KB | Training curves |

---

## ✅ Summary

After running all three scripts (`train.py`, `test.py`, `validation.py`):

1. ✓ Model trained and saved to `models/best_mobilenetv2.keras`
2. ✓ Training metrics saved to `models/training_metrics.txt`
3. ✓ Test metrics saved to `models/test_metrics.txt`
4. ✓ Validation metrics saved to `models/validation_metrics.txt`
5. ✓ Training curves saved to `models/training_curves.png`
6. ✓ Class names saved to `models/class_names.json`

All metrics and accuracy data are stored as **text files** in the `models` folder for easy review and record-keeping!
