"""
Generate a DOCX step-by-step guide for the Cattle Disease Detection project
"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE

def create_guide():
    doc = Document()

    # Title
    title = doc.add_heading('Cattle Disease Detection System - Step-by-Step Guide', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Introduction
    doc.add_heading('What This Project Does', level=1)
    doc.add_paragraph(
        'This is an AI-powered system that automatically detects diseases in cattle from images. '
        'It can identify:'
    )
    diseases = [
        'Dermatophilosis - Bacterial skin infection',
        'FMD (Foot and Mouth Disease) - Viral disease affecting mouth & hooves',
        'Lumpy Skin - Viral skin disease',
        'Healthy - Normal cattle (control class)'
    ]
    for disease in diseases:
        p = doc.add_paragraph(disease, style='List Bullet')

    doc.add_paragraph(
        'The system uses 4 different AI models working together (ensemble) for high accuracy (~97-99%).'
    )

    # Quick Start
    doc.add_heading('Quick Start (If You Just Want to Use It)', level=1)
    doc.add_paragraph('If you already have the trained models and just want to predict diseases:')

    doc.add_paragraph('1. Open Command Prompt/Terminal', style='List Number')
    doc.add_paragraph('2. Navigate to the project folder:', style='List Number')
    code = doc.add_paragraph('cd E:\\CODE')
    code.style = 'Intense Quote'

    doc.add_paragraph('3. Run prediction on an image:', style='List Number')
    code = doc.add_paragraph('python ensemble_predict.py "path\\to\\your\\image.jpg"')
    code.style = 'Intense Quote'

    doc.add_paragraph("That's it! The system will tell you what disease (if any) is detected.")

    # Complete Guide
    doc.add_page_break()
    doc.add_heading('Complete Step-by-Step Guide', level=1)

    # Step 1
    doc.add_heading('STEP 1: Check Your Computer Requirements', level=2)
    doc.add_heading('Minimum Requirements:', level=3)
    reqs = [
        'Windows 10 or 11',
        'At least 8GB RAM (16GB recommended)',
        '5GB free disk space',
        'Internet connection (for installing software)'
    ]
    for req in reqs:
        doc.add_paragraph(req, style='List Bullet')

    doc.add_heading('Optional but Recommended:', level=3)
    optional = [
        'NVIDIA GPU with CUDA support (for faster processing)',
        '16GB+ RAM'
    ]
    for opt in optional:
        doc.add_paragraph(opt, style='List Bullet')

    # Step 2
    doc.add_heading('STEP 2: Install Python', level=2)
    doc.add_paragraph('1. Download Python from: https://www.python.org/downloads/')
    doc.add_paragraph('2. Download Python 3.10 or 3.11 (recommended)')
    doc.add_paragraph('3. Run the installer')
    doc.add_paragraph('4. IMPORTANT: Check the box "Add Python to PATH" during installation')
    doc.add_paragraph('5. Click "Install Now"')

    doc.add_heading('Verify Python is installed:', level=3)
    code = doc.add_paragraph('python --version')
    code.style = 'Intense Quote'
    doc.add_paragraph('You should see something like: Python 3.10.x or Python 3.11.x')

    # Step 3
    doc.add_heading('STEP 3: Install Required Libraries', level=2)
    doc.add_paragraph('Open Command Prompt and run:')

    code = doc.add_paragraph('pip install tensorflow scikit-learn matplotlib seaborn opencv-python numpy')
    code.style = 'Intense Quote'

    doc.add_paragraph('Or use specific versions (more stable):')
    code = doc.add_paragraph('pip install tensorflow==2.15.0 scikit-learn==1.3.2 matplotlib==3.8.2 seaborn==0.13.0 opencv-python==4.8.1 numpy==1.24.3')
    code.style = 'Intense Quote'

    doc.add_heading('What each library does:', level=3)
    libs = [
        'tensorflow - AI/Deep learning framework',
        'scikit-learn - Machine learning utilities',
        'matplotlib - Plotting graphs',
        'seaborn - Statistical visualizations',
        'opencv-python - Image processing',
        'numpy - Numerical computations'
    ]
    for lib in libs:
        doc.add_paragraph(lib, style='List Bullet')

    # Step 4
    doc.add_heading('STEP 4: Understand the Project Structure', level=2)
    doc.add_paragraph('The project is organized as follows:')

    code = doc.add_paragraph('''E:\\CODE\\
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
└── requirements.txt        ← List of required libraries''')
    code.style = 'Intense Quote'

    # Step 5
    doc.add_heading('STEP 5: Prepare Your Dataset (If Training New Models)', level=2)
    doc.add_paragraph('If you want to train your own models, organize your images like this:')

    code = doc.add_paragraph('''dataset_augmented/
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
    └── lumpy skin/''')
    code.style = 'Intense Quote'

    doc.add_heading('Requirements:', level=3)
    dataset_reqs = [
        'Minimum 10 images per class per split',
        'Recommended 50+ images per class per split',
        'Supported formats: .jpg, .jpeg, .png',
        'Image size: Any size (will be resized automatically)'
    ]
    for req in dataset_reqs:
        doc.add_paragraph(req, style='List Bullet')

    # Step 6
    doc.add_heading('STEP 6: Train Individual Models (Optional)', level=2)
    doc.add_paragraph('Note: The models in the folders are already trained. Skip this step if you just want to use them.')

    doc.add_heading('To Train MobileNetV2:', level=3)
    code = doc.add_paragraph('cd E:\\CODE\\MBV2\npython train.py')
    code.style = 'Intense Quote'

    doc.add_heading('To Train EfficientNetB0:', level=3)
    code = doc.add_paragraph('cd E:\\CODE\\EFFNETB0\npython train.py')
    code.style = 'Intense Quote'

    doc.add_heading('To Train DenseNet201:', level=3)
    code = doc.add_paragraph('cd E:\\CODE\\DENSENT201\npython train.py')
    code.style = 'Intense Quote'

    doc.add_heading('To Train ResNet101:', level=3)
    code = doc.add_paragraph('cd E:\\CODE\\RESNET101\npython train.py')
    code.style = 'Intense Quote'

    doc.add_heading('Training Time:', level=3)
    doc.add_paragraph('• With GPU: 25-35 minutes per model', style='List Bullet')
    doc.add_paragraph('• Without GPU: 1.5-2 hours per model', style='List Bullet')

    # Step 7
    doc.add_heading('STEP 7: Test Individual Models (Optional)', level=2)
    doc.add_paragraph('After training, test each model:')

    doc.add_paragraph('For MobileNetV2:', style='List Bullet')
    code = doc.add_paragraph('cd E:\\CODE\\MBV2\npython test.py')
    code.style = 'Intense Quote'

    doc.add_paragraph('For EfficientNetB0:', style='List Bullet')
    code = doc.add_paragraph('cd E:\\CODE\\EFFNETB0\npython test.py')
    code.style = 'Intense Quote'

    doc.add_paragraph('For DenseNet201:', style='List Bullet')
    code = doc.add_paragraph('cd E:\\CODE\\DENSENT201\npython test.py')
    code.style = 'Intense Quote'

    doc.add_paragraph('For ResNet101:', style='List Bullet')
    code = doc.add_paragraph('cd E:\\CODE\\RESNET101\npython test.py')
    code.style = 'Intense Quote'

    doc.add_paragraph('This will generate:')
    doc.add_paragraph('• confusion_matrix.png - Visual accuracy chart', style='List Bullet')
    doc.add_paragraph('• classification_report.txt - Detailed metrics', style='List Bullet')

    # Step 8
    doc.add_page_break()
    doc.add_heading('STEP 8: Use the Ensemble System (Recommended)', level=2)
    doc.add_paragraph('The ensemble system combines all 4 models for better accuracy.')

    doc.add_heading('8.1 Predict on a Single Image', level=3)
    code = doc.add_paragraph('cd E:\\CODE\npython ensemble_predict.py "path\\to\\your\\image.jpg"')
    code.style = 'Intense Quote'

    doc.add_heading('Example:', level=3)
    code = doc.add_paragraph('python ensemble_predict.py "dataset_augmented\\test\\healthy\\cow1.jpg"')
    code.style = 'Intense Quote'

    doc.add_paragraph('Output will show:')
    doc.add_paragraph('• Individual predictions from each model', style='List Bullet')
    doc.add_paragraph('• Model voting results', style='List Bullet')
    doc.add_paragraph('• Final ensemble prediction with confidence', style='List Bullet')

    doc.add_heading('8.2 Save Prediction Results', level=3)
    code = doc.add_paragraph('python ensemble_predict.py "path\\to\\image.jpg" --save-json result.json')
    code.style = 'Intense Quote'

    doc.add_heading('8.3 Batch Process Multiple Images', level=3)
    code = doc.add_paragraph('python batch_ensemble_predict.py "path\\to\\image_folder"')
    code.style = 'Intense Quote'

    doc.add_heading('Example:', level=3)
    code = doc.add_paragraph('python batch_ensemble_predict.py "dataset_augmented\\test"')
    code.style = 'Intense Quote'

    doc.add_paragraph('This will process all images and generate:')
    doc.add_paragraph('• batch_results_[timestamp].json - Detailed results', style='List Bullet')
    doc.add_paragraph('• batch_results_[timestamp].csv - Spreadsheet format', style='List Bullet')
    doc.add_paragraph('• batch_summary_[timestamp].json - Statistics', style='List Bullet')

    doc.add_heading('8.4 Verify System Setup', level=3)
    code = doc.add_paragraph('python quickstart_ensemble.py check')
    code.style = 'Intense Quote'
    doc.add_paragraph('This checks if all required files are present.')

    # Step 9
    doc.add_heading('STEP 9: Understanding the Results', level=2)

    doc.add_heading('High Confidence (Trust It!)', level=3)
    doc.add_paragraph('• Confidence ≥ 90%', style='List Bullet')
    doc.add_paragraph('• All 4 models agree (4/4 votes)', style='List Bullet')
    doc.add_paragraph('• Example: "FMD - 96.34% confidence"', style='List Bullet')

    doc.add_heading('Moderate Confidence (Review It)', level=3)
    doc.add_paragraph('• Confidence 75-90%', style='List Bullet')
    doc.add_paragraph('• 3/4 or 2/4 models agree', style='List Bullet')
    doc.add_paragraph('• Consider reviewing the image manually', style='List Bullet')

    doc.add_heading('Low Confidence (Investigate)', level=3)
    doc.add_paragraph('• Confidence < 75%', style='List Bullet')
    doc.add_paragraph('• Models disagree', style='List Bullet')
    doc.add_paragraph('• Image may be unclear or borderline case', style='List Bullet')

    # Troubleshooting
    doc.add_page_break()
    doc.add_heading('Troubleshooting', level=1)

    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'

    # Header row
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Problem'
    hdr_cells[1].text = 'Solution'

    # Data rows
    problems = [
        ('Python is not recognized', 'Reinstall Python\nMake sure to check "Add Python to PATH"\nRestart Command Prompt'),
        ('pip is not recognized', 'Try: python -m pip install ...\nOr reinstall Python with PATH option'),
        ('CUDA not available (GPU not working)', 'This is not critical. The system will use CPU (slower but works)\nTo enable GPU, install NVIDIA drivers and CUDA toolkit'),
        ('Out of Memory (OOM)', 'Close other programs\nReduce batch size in train.py (change 32 to 16)\nUse smaller images'),
        ('Model file not found', 'Check that .keras files exist in model folders\nVerify you\'re in the correct directory\nCheck file paths are correct'),
        ('Could not read image', 'Check image file is not corrupted\nTry a different image format (JPG, PNG)\nVerify the file path is correct')
    ]

    for problem, solution in problems:
        row = table.add_row()
        row.cells[0].text = problem
        row.cells[1].text = solution

    # Quick Reference
    doc.add_page_break()
    doc.add_heading('Quick Reference Commands', level=1)

    table = doc.add_table(rows=1, cols=2)
    table.style = 'Table Grid'

    # Header row
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Task'
    hdr_cells[1].text = 'Command'

    # Data rows
    commands = [
        ('Navigate to project', 'cd E:\\CODE'),
        ('Check Python version', 'python --version'),
        ('Install dependencies', 'pip install -r requirements.txt'),
        ('Predict single image', 'python ensemble_predict.py "image.jpg"'),
        ('Batch process folder', 'python batch_ensemble_predict.py "folder"'),
        ('Check system setup', 'python quickstart_ensemble.py check'),
        ('Train MobileNetV2', 'cd MBV2 && python train.py'),
        ('Test MobileNetV2', 'cd MBV2 && python test.py')
    ]

    for task, cmd in commands:
        row = table.add_row()
        row.cells[0].text = task
        row.cells[1].text = cmd

    # What Each Script Does
    doc.add_heading('What Each Script Does', level=1)

    table = doc.add_table(rows=1, cols=3)
    table.style = 'Table Grid'

    # Header row
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Script'
    hdr_cells[1].text = 'Purpose'
    hdr_cells[2].text = 'Location'

    # Data rows
    scripts = [
        ('train.py', 'Train a model', 'Each model folder'),
        ('test.py', 'Evaluate model on test set', 'Each model folder'),
        ('validation.py', 'Validate on validation set', 'Each model folder'),
        ('predict.py', 'Predict single image', 'Each model folder'),
        ('ensemble_predict.py', 'Use all 4 models together', 'E:\\CODE\\')
    ]

    for script, purpose, location in scripts:
        row = table.add_row()
        row.cells[0].text = script
        row.cells[1].text = purpose
        row.cells[2].text = location

    # Tips
    doc.add_page_break()
    doc.add_heading('Tips for Best Results', level=1)

    doc.add_heading('1. Use Good Quality Images', level=2)
    tips1 = [
        'Clear, well-lit photos',
        'Focus on affected areas',
        'Avoid blurry images'
    ]
    for tip in tips1:
        doc.add_paragraph(tip, style='List Bullet')

    doc.add_heading('2. Proper Dataset Organization', level=2)
    tips2 = [
        'Keep class folders separate',
        'Use consistent naming',
        'Balance classes (similar number of images per disease)'
    ]
    for tip in tips2:
        doc.add_paragraph(tip, style='List Bullet')

    doc.add_heading('3. Training Tips', level=2)
    tips3 = [
        'Start with small dataset to test',
        'Monitor training curves',
        'Use GPU if available'
    ]
    for tip in tips3:
        doc.add_paragraph(tip, style='List Bullet')

    doc.add_heading('4. Prediction Tips', level=2)
    tips4 = [
        'Use ensemble system for best accuracy',
        'Review low-confidence predictions',
        'Keep human expert in the loop'
    ]
    for tip in tips4:
        doc.add_paragraph(tip, style='List Bullet')

    # How It Works
    doc.add_page_break()
    doc.add_heading('How It Works (Simple Explanation)', level=1)

    steps = [
        ('1. Input', 'You provide an image of a cow'),
        ('2. Processing', 'The image is resized and prepared'),
        ('3. Analysis', '4 different AI models analyze the image'),
        ('4. Voting', 'Each model "votes" on what disease it sees'),
        ('5. Averaging', 'Confidence scores are averaged'),
        ('6. Output', 'Final prediction with confidence percentage')
    ]

    for step_num, (title, desc) in enumerate(steps):
        doc.add_heading(title, level=2)
        doc.add_paragraph(desc)

    doc.add_heading('Example:', level=2)
    code = doc.add_paragraph('''Image: cow_with_lesions.jpg

MobileNetV2 says: Lumpy Skin (92%)
EfficientNetB0 says: Lumpy Skin (89%)
DenseNet201 says: Lumpy Skin (94%)
ResNet101 says: Lumpy Skin (91%)

Final: LUMPY SKIN (91.5% confidence)''')
    code.style = 'Intense Quote'

    # Next Steps
    doc.add_page_break()
    doc.add_heading('Next Steps', level=1)

    doc.add_heading('1. Try it out:', level=2)
    code = doc.add_paragraph('cd E:\\CODE\npython ensemble_predict.py "dataset_augmented\\test\\healthy\\some_image.jpg"')
    code.style = 'Intense Quote'

    doc.add_heading('2. Batch process your images:', level=2)
    code = doc.add_paragraph('python batch_ensemble_predict.py "your_image_folder"')
    code.style = 'Intense Quote'

    doc.add_heading('3. Review results:', level=2)
    review = [
        'Open the generated CSV file in Excel',
        'Check confidence scores',
        'Verify predictions match expectations'
    ]
    for item in review:
        doc.add_paragraph(item, style='List Bullet')

    # Important Notes
    doc.add_page_break()
    doc.add_heading('Important Notes', level=1)

    notes = [
        'This system is for educational and research purposes',
        'Always have a veterinarian verify diagnoses',
        'The system is a decision support tool, not a replacement for expert judgment',
        'Regular updates and retraining may be needed for new disease variants'
    ]
    for note in notes:
        doc.add_paragraph(note, style='List Bullet')

    # Checklist
    doc.add_heading('Checklist Before Starting', level=1)

    checklist = [
        'Python 3.10+ installed',
        'All required libraries installed',
        'Dataset organized correctly',
        'Model files present (or training completed)',
        'Enough disk space available',
        'Command Prompt/Terminal working'
    ]
    for item in checklist:
        doc.add_paragraph(item, style='List Bullet')

    # Footer
    doc.add_page_break()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('Version: 1.0 | Last Updated: May 2026 | Status: Production Ready ✓')
    run.bold = True

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('Good luck with your cattle disease detection project! 🐄🔬')
    run.bold = True

    # Save the document
    output_path = r'E:\CODE\CATTLE_DISEASE_DETECTION_GUIDE.docx'
    doc.save(output_path)
    print(f'Document created successfully: {output_path}')

if __name__ == '__main__':
    create_guide()
