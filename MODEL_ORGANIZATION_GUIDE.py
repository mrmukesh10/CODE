"""
MODEL STRUCTURE ORGANIZATION GUIDE
All models follow the same organized structure:

├── Each Model Folder (densenet201, efficientnetb0, resnet101, mobilenetv2)
│   ├── train.py              # Training script (Phase 1 + Phase 2)
│   ├── test.py               # Test evaluation
│   ├── validation.py         # Validation evaluation
│   ├── predict.py            # Single image prediction ✨ NEW
│   └── results/              # All outputs stored here
│       ├── training_history.png        # Training curves
│       ├── confusion_matrix.png        # Test confusion matrix
│       ├── validation_confusion_matrix.png  # Validation confusion matrix
│       ├── confidence_distribution.png # Confidence histogram
│       ├── test_metrics.json           # Test metrics
│       ├── validation_metrics.json     # Validation metrics
│       └── prediction_result.png       # Single image prediction results
│
├── models/                   # Shared model storage
│   ├── best_densenet201.keras
│   ├── best_efficientnetb0.keras
│   ├── best_resnet101.keras
│   ├── best_mobilenetv2.keras
│   ├── class_names.json
│   ├── temperature.json
│   └── ... (other shared files)
│
└── ensemble_predict.py       # Ensemble prediction from all 4 models
"""

# QUICK COMMAND REFERENCE
# =====================

# 1. TRAIN A MODEL
# ----------------
# python d:\project\densenet201\train.py
# python d:\project\efficientnetb0\train.py
# python d:\project\resnet101\train.py
# python d:\project\mobilenetv2\train.py

# 2. TEST A MODEL
# ---------------
# python d:\project\densenet201\test.py
# python d:\project\efficientnetb0\test.py
# python d:\project\resnet101\test.py
# python d:\project\mobilenetv2\test.py

# 3. VALIDATE A MODEL
# -------------------
# python d:\project\densenet201\validation.py
# python d:\project\efficientnetb0\validation.py
# python d:\project\resnet101\validation.py
# python d:\project\mobilenetv2\validation.py

# 4. PREDICT WITH SINGLE MODEL
# ----------------------------
# python d:\project\densenet201\predict.py "C:\path\to\image.jpg"
# python d:\project\efficientnetb0\predict.py "C:\path\to\image.jpg"
# python d:\project\resnet101\predict.py "C:\path\to\image.jpg"
# python d:\project\mobilenetv2\predict.py "C:\path\to\image.jpg"

# 5. ENSEMBLE PREDICTION (All 4 Models)
# ------------------------------------
# python d:\project\ensemble_predict.py "C:\path\to\image.jpg"

# OUTPUT LOCATIONS
# ================

# Model Files → D:\project\models\
# DenseNet201 Results → D:\project\densenet201\results\
# EfficientNetB0 Results → D:\project\efficientnetb0\results\
# ResNet101 Results → D:\project\resnet101\results\
# MobileNetV2 Results → D:\project\mobilenetv2\results\
# Ensemble Results → D:\project\mobilenetv2\results\ensemble_*

# FILES IN RESULTS FOLDERS
# ========================

# training_history.png        → Training accuracy & loss curves
# confusion_matrix.png        → Test set confusion matrix
# validation_confusion_matrix.png → Validation set confusion matrix  
# confidence_distribution.png → Prediction confidence histogram
# test_metrics.json           → Test accuracy, loss, metrics
# validation_metrics.json     → Validation accuracy, loss, metrics
# prediction_result.png       → Single image prediction visualization
# prediction_result.json      → Single image prediction details
# ensemble_prediction.png     → Ensemble prediction comparison (all 4 models)
# ensemble_prediction.json    → Detailed ensemble results

print(__doc__)
