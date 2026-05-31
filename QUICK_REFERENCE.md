# 🚀 Quick Reference Card - Cattle Disease Detection

## 📍 Navigate to Project
```bash
cd E:\CODE
```

## 🔧 Setup Commands
```bash
# Check Python version
python --version

# Install all dependencies
pip install -r requirements.txt

# Install specific versions
pip install tensorflow==2.15.0 scikit-learn==1.3.2 matplotlib==3.8.2 seaborn==0.13.0 opencv-python==4.8.1 numpy==1.24.3
```

## 🎯 Prediction Commands

### Single Image Prediction
```bash
python ensemble_predict.py "path\to\image.jpg"
```

### Save Results to JSON
```bash
python ensemble_predict.py "image.jpg" --save-json result.json
```

### Batch Process Folder
```bash
python batch_ensemble_predict.py "path\to\folder"
```

### Custom Output Directory
```bash
python batch_ensemble_predict.py "folder" --output-dir "results"
```

## 🧪 Testing Commands

### Check System Setup
```bash
python quickstart_ensemble.py check
```

### Run Demo
```bash
python quickstart_ensemble.py demo
```

### Show Usage Guide
```bash
python quickstart_ensemble.py guide
```

## 🏋️ Training Commands (Optional)

### Train MobileNetV2
```bash
cd MBV2
python train.py
```

### Train EfficientNetB0
```bash
cd EFFNETB0
python train.py
```

### Train DenseNet201
```bash
cd DENSENT201
python train.py
```

### Train ResNet101
```bash
cd RESNET101
python train.py
```

## 📊 Evaluation Commands

### Test MobileNetV2
```bash
cd MBV2
python test.py
```

### Test EfficientNetB0
```bash
cd EFFNETB0
python test.py
```

### Test DenseNet201
```bash
cd DENSENT201
python test.py
```

### Test ResNet101
```bash
cd RESNET101
python test.py
```

## 📁 Project Structure

```
E:\CODE\
├── MBV2/              ← MobileNetV2 (32MB model)
├── EFFNETB0/          ← EfficientNetB0 (40MB model)
├── DENSENT201/        ← DenseNet201 (135MB model)
├── RESNET101/         ← ResNet101 (361MB model)
├── ensemble_predict.py      ← Main prediction script
├── batch_ensemble_predict.py  ← Batch processing
├── dataset_augmented/  ← Your images
└── requirements.txt    ← Dependencies
```

## 🎯 Detectable Diseases

| Disease | Description |
|---------|-------------|
| Dermatophilosis | Bacterial skin infection |
| FMD | Foot and Mouth Disease |
| Lumpy Skin | Viral skin disease |
| Healthy | Normal cattle |

## ⚡ Confidence Levels

| Confidence | Action |
|------------|--------|
| ≥ 90% | Trust it! |
| 75-90% | Review it |
| < 75% | Investigate |

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Python not found | Reinstall with PATH option |
| pip not found | Use `python -m pip` |
| CUDA not available | Use CPU (slower but works) |
| Out of Memory | Close other programs |
| Model not found | Check file paths |

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Accuracy | 97-99% |
| Processing Time | 2-3 sec (GPU) / 10-15 sec (CPU) |
| Memory Usage | ~2GB |

---

**For detailed guide, see: STEP_BY_STEP_GUIDE.md**
