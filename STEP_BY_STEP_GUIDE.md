# 🐄 Cattle Disease Detection System - Step-by-Step Guide

## 📋 What This Project Does

This is an **AI-powered system** that automatically detects diseases in cattle from images. It can identify:

- **Dermatophilosis** - Bacterial skin infection
- **FMD** (Foot and Mouth Disease) - Viral disease affecting mouth & hooves
- **Lumpy Skin** - Viral skin disease
- **Healthy** - Normal cattle (control class)

The system uses **4 different AI models** working together (ensemble) for high accuracy (~97-99%).

---

## 🎯 Quick Start (If You Just Want to Use It)

If you already have the trained models and just want to predict diseases:

1. Open Command Prompt/Terminal
2. Navigate to the project folder:
   ```bash
   cd E:\CODE
   ```
3. Run prediction on an image:
   ```bash
   python ensemble_predict.py "path\to\your\image.jpg"
   ```

That's it! The system will tell you what disease (if any) is detected.

---

## 📖 Complete Step-by-Step Guide

### STEP 1: Check Your Computer Requirements

**Minimum Requirements:**
- Windows 10 or 11
- At least 8GB RAM (16GB recommended)
- 5GB free disk space
- Internet connection (for installing software)

**Optional but Recommended:**
- NVIDIA GPU with CUDA support (for faster processing)
- 16GB+ RAM

---

### STEP 2: Install Python

1. Download Python from: https://www.python.org/downloads/
2. Download **Python 3.10 or 3.11** (recommended)
3. Run the installer
4. **IMPORTANT:** Check the box "Add Python to PATH" during installation
5. Click "Install Now"

**Verify Python is installed:**
```bash
python --version
```

You should see something like: `Python 3.10.x` or `Python 3.11.x`

---

### STEP 3: Install Required Libraries

Open Command Prompt and run:

```bash
pip install tensorflow scikit-learn matplotlib seaborn opencv-python numpy
```

**Or use specific versions (more stable):**
```bash
pip install tensorflow==2.15.0 scikit-learn==1.3.2 matplotlib==3.8.2 seaborn==0.13.0 opencv-python==4.8.1 numpy==1.24.3
```

**What each library does:**
- `tensorflow` - AI/Deep learning framework
- `scikit-learn` - Machine learning utilities
- `matplotlib` - Plotting graphs
- `seaborn` - Statistical visualizations
- `opencv-python` - Image processing
- `numpy` - Numerical computations

---

### STEP 4: Understand the Project Structure

```
E:\CODE\
├── MBV2/                    ← MobileNetV2 model folder
│   ├── train.py            ← Training script
│   ├── test.py             ← Testing script
│   ├── validation.py       ← Validation script
│   ├── predict.py          ← Single image prediction
│   └── best_mobilenetv2.keras  ← Trained model (32MB)
│
├── EFFNETB0/               ← EfficientNetB0 model folder
│   ├── train.py
│   ├── test.py
│   ├── validation.py
│   ├── predict.py
│   └── best_efficientnetb0.keras  ← Trained model (40MB)
│
├── DENSENT201/             ← DenseNet201 model folder
│   ├── train.py
│   ├── test.py
│   ├── validation.py
│   ├── predict.py
│   └── best_densenet201.keras    ← Trained model (135MB)
│
├── RESNET101/              ← ResNet101 model folder
│   ├── train.py
│   ├── test.py
│   ├── validation.py
│   ├── predict.py
│   └── best_resnet101.keras      ← Trained model (361MB)
│
├── ensemble_predict.py      ← Main script (uses all 4 models)
├── dataset_augmented/      ← Your dataset folder
│   ├── train/              ← Training images
│   ├── valid/              ← Validation images
│   └── test/               ← Test images
│
└── requirements.txt        ← List of required libraries
```

---

### STEP 5: Prepare Your Dataset (If Training New Models)

If you want to train your own models, organize your images like this:

```
dataset_augmented/
├── train/                    (60-70% of your images)
│   ├── dermatophilosis/      ← Put dermatophilosis images here
│   ├── fmd/                  ← Put FMD images here
│   ├── healthy/              ← Put healthy cattle images here
│   └── lumpy skin/           ← Put lumpy skin images here
│
├── valid/                    (15-20% of your images)
│   ├── dermatophilosis/
│   ├── fmd/
│   ├── healthy/
│   └── lumpy skin/
│
└── test/                     (15-20% of your images)
    ├── dermatophilosis/
    ├── fmd/
    ├── healthy/
    └── lumpy skin/
```

**Requirements:**
- Minimum 10 images per class per split
- Recommended 50+ images per class per split
- Supported formats: `.jpg`, `.jpeg`, `.png`
- Image size: Any size (will be resized automatically)

---

### STEP 6: Train Individual Models (Optional)

**Note:** The models in the folders are already trained. Skip this step if you just want to use them.

#### To Train MobileNetV2:

```bash
cd E:\CODE\MBV2
python train.py
```

#### To Train EfficientNetB0:

```bash
cd E:\CODE\EFFNETB0
python train.py
```

#### To Train DenseNet201:

```bash
cd E:\CODE\DENSENT201
python train.py
```

#### To Train ResNet101:

```bash
cd E:\CODE\RESNET101
python train.py
```

**Training Time:**
- With GPU: 25-35 minutes per model
- Without GPU: 1.5-2 hours per model

---

### STEP 7: Test Individual Models (Optional)

After training, test each model:

```bash
# For MobileNetV2
cd E:\CODE\MBV2
python test.py

# For EfficientNetB0
cd E:\CODE\EFFNETB0
python test.py

# For DenseNet201
cd E:\CODE\DENSENT201
python test.py

# For ResNet101
cd E:\CODE\RESNET101
python test.py
```

This will generate:
- `confusion_matrix.png` - Visual accuracy chart
- `classification_report.txt` - Detailed metrics

---

### STEP 8: Use the Ensemble System (Recommended)

The ensemble system combines all 4 models for better accuracy.

#### 8.1 Predict on a Single Image

```bash
cd E:\CODE
python ensemble_predict.py "path\to\your\image.jpg"
```

**Example:**
```bash
python ensemble_predict.py "dataset_augmented\test\healthy\cow1.jpg"
```

**Output will show:**
- Individual predictions from each model
- Model voting results
- Final ensemble prediction with confidence

#### 8.2 Save Prediction Results

```bash
python ensemble_predict.py "path\to\image.jpg" --save-json result.json
```

#### 8.3 Batch Process Multiple Images

```bash
python batch_ensemble_predict.py "path\to\image_folder"
```

**Example:**
```bash
python batch_ensemble_predict.py "dataset_augmented\test"
```

This will process all images and generate:
- `batch_results_[timestamp].json` - Detailed results
- `batch_results_[timestamp].csv` - Spreadsheet format
- `batch_summary_[timestamp].json` - Statistics

#### 8.4 Verify System Setup

```bash
python quickstart_ensemble.py check
```

This checks if all required files are present.

---

### STEP 9: Understanding the Results

#### High Confidence (Trust It!)
- Confidence ≥ 90%
- All 4 models agree (4/4 votes)
- Example: "FMD - 96.34% confidence"

#### Moderate Confidence (Review It)
- Confidence 75-90%
- 3/4 or 2/4 models agree
- Consider reviewing the image manually

#### Low Confidence (Investigate)
- Confidence < 75%
- Models disagree
- Image may be unclear or borderline case

---

## 🔧 Troubleshooting

### Problem: "Python is not recognized"
**Solution:**
1. Reinstall Python
2. Make sure to check "Add Python to PATH"
3. Restart Command Prompt

### Problem: "pip is not recognized"
**Solution:**
1. Try: `python -m pip install ...`
2. Or reinstall Python with PATH option

### Problem: "CUDA not available" (GPU not working)
**Solution:**
- This is not critical. The system will use CPU (slower but works)
- To enable GPU, install NVIDIA drivers and CUDA toolkit

### Problem: "Out of Memory" (OOM)
**Solution:**
1. Close other programs
2. Reduce batch size in train.py (change 32 to 16)
3. Use smaller images

### Problem: "Model file not found"
**Solution:**
1. Check that `.keras` files exist in model folders
2. Verify you're in the correct directory
3. Check file paths are correct

### Problem: "Could not read image"
**Solution:**
1. Check image file is not corrupted
2. Try a different image format (JPG, PNG)
3. Verify the file path is correct

---

## 📊 Quick Reference Commands

| Task | Command |
|------|---------|
| Navigate to project | `cd E:\CODE` |
| Check Python version | `python --version` |
| Install dependencies | `pip install -r requirements.txt` |
| Predict single image | `python ensemble_predict.py "image.jpg"` |
| Batch process folder | `python batch_ensemble_predict.py "folder"` |
| Check system setup | `python quickstart_ensemble.py check` |
| Train MobileNetV2 | `cd MBV2 && python train.py` |
| Test MobileNetV2 | `cd MBV2 && python test.py` |

---

## 🎯 What Each Script Does

| Script | Purpose | Location |
|--------|---------|----------|
| `train.py` | Train a model | Each model folder |
| `test.py` | Evaluate model on test set | Each model folder |
| `validation.py` | Validate on validation set | Each model folder |
| `predict.py` | Predict single image | Each model folder |
| `ensemble_predict.py` | Use all 4 models together | E:\CODE\ |
| `batch_ensemble_predict.py` | Process multiple images | E:\CODE\ |

---

## 💡 Tips for Best Results

1. **Use Good Quality Images**
   - Clear, well-lit photos
   - Focus on affected areas
   - Avoid blurry images

2. **Proper Dataset Organization**
   - Keep class folders separate
   - Use consistent naming
   - Balance classes (similar number of images per disease)

3. **Training Tips**
   - Start with small dataset to test
   - Monitor training curves
   - Use GPU if available

4. **Prediction Tips**
   - Use ensemble system for best accuracy
   - Review low-confidence predictions
   - Keep human expert in the loop

---

## 📞 Getting Help

If you encounter issues:

1. Check the error message carefully
2. Verify all dependencies are installed
3. Ensure file paths are correct
4. Check that model files exist

---

## ✅ Checklist Before Starting

- [ ] Python 3.10+ installed
- [ ] All required libraries installed
- [ ] Dataset organized correctly
- [ ] Model files present (or training completed)
- [ ] Enough disk space available
- [ ] Command Prompt/Terminal working

---

## 🎓 How It Works (Simple Explanation)

1. **Input:** You provide an image of a cow
2. **Processing:** The image is resized and prepared
3. **Analysis:** 4 different AI models analyze the image
4. **Voting:** Each model "votes" on what disease it sees
5. **Averaging:** Confidence scores are averaged
6. **Output:** Final prediction with confidence percentage

**Example:**
```
Image: cow_with_lesions.jpg

MobileNetV2 says: Lumpy Skin (92%)
EfficientNetB0 says: Lumpy Skin (89%)
DenseNet201 says: Lumpy Skin (94%)
ResNet101 says: Lumpy Skin (91%)

Final: LUMPY SKIN (91.5% confidence)
```

---

## 🚀 Next Steps

1. **Try it out:**
   ```bash
   cd E:\CODE
   python ensemble_predict.py "dataset_augmented\test\healthy\some_image.jpg"
   ```

2. **Batch process your images:**
   ```bash
   python batch_ensemble_predict.py "your_image_folder"
   ```

3. **Review results:**
   - Open the generated CSV file in Excel
   - Check confidence scores
   - Verify predictions match expectations

---

## 📝 Notes

- This system is for **educational and research purposes**
- Always have a veterinarian verify diagnoses
- The system is a **decision support tool**, not a replacement for expert judgment
- Regular updates and retraining may be needed for new disease variants

---

**Version:** 1.0
**Last Updated:** May 2026
**Status:** Production Ready ✓

---

**Good luck with your cattle disease detection project! 🐄🔬**
