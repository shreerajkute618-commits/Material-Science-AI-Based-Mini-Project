# AI-Based Identification and Sorting of Recyclable Materials ♻️
### Materials Science and Engineering Laboratory Mini-Project

An end-to-end, ready-to-run computer vision and deep learning application that identifies post-consumer materials from digital imagery or live webcam feeds, maps them to fundamental Materials Science properties, and simulates automated sorting in a Material Recovery Facility (MRF).

---

## 🌟 Highlights & Key Features

- **Genuine AI Classification**: Pretrained **MobileNetV2** deep convolutional neural network fine-tuned using Transfer Learning on the benchmark Stanford TrashNet dataset (2,527 real-world images).
- **6 Circular Economy Categories**:
  1. **Plastic** (PET, HDPE, PP, etc.)
  2. **Metal** (Aluminum & Steels)
  3. **Paper** (Cellulose fibers)
  4. **Cardboard** (Corrugated fiberboard)
  5. **Glass** (Soda-lime silicate)
  6. **Other / Non-recyclable waste** (Multi-material laminates & refuse)
- **Materials Science & Engineering Integration**:
  - Chemical composition & microstructure analysis (macromolecular chains, crystalline metallic lattice, amorphous silica network).
  - Physical & mechanical properties (density, melting point, thermal conductivity, tensile strength).
  - Circular lifecycle analysis, recycling thermodynamics, and facility best practices.
- **Uncertainty & Safety Guardrails**: Calibrated Softmax probability distribution with automatic detection of low-confidence predictions (<50%) to prevent overconfident sorting errors.
- **Interactive Web Interface**: Streamlit dashboard supporting file uploads (JPG, PNG, WEBP), browser webcam capture, probability distribution charts, and session history.
- **Digital Twin Sorting Simulation**: Visual multi-chute conveyor and pneumatic diverter line demonstrating industrial MRF automation.
- **Persistent Session Logging**: Automatic history logging with one-click CSV export for facility waste auditing.
- **Academic Documentation & Viva Prep**: Complete project report (Markdown & DOCX), 10-slide presentation (PPTX & Markdown), 2-minute demo script, and 25+ viva voce questions and answers.

---

## 🏛️ Project Architecture

```
                                  [ User Input: File / Webcam ]
                                                │
                                                ▼
                                    [ Image Preprocessing ]
                                    • Resize to 224x224 RGB
                                    • ImageNet Normalization
                                                │
                                                ▼
                                     [ MobileNetV2 Deep CNN ]
                                    • Inverted Residual Blocks
                                    • Depthwise Separable Conv
                                                │
                                                ▼
                                      [ Softmax Classifier ]
                                     • 6 Target Class Scores
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
     [ Uncertainty Guardrail ]     [ Materials Science Engine ]    [ Conveyor Sorting Twin ]
     • Flag if Top Prob < 50%       • Microstructure & Bonding      • Chute Selection
     • Warn against ambiguities     • Physical / Thermal Specs      • Actuator Trigger Logic
                 │                              │                              │
                 └──────────────────────────────┼──────────────────────────────┘
                                                │
                                                ▼
                                   [ Persistent History Logger ]
                                    • prediction_history.csv
                                    • One-Click CSV Export
```

---

## 📁 Project Folder Structure

```
Material Science AI Based Mini Project/
├── app.py                      # Main interactive Streamlit web application
├── config.py                   # Centralized configuration & Materials Science data
├── download_dataset.py         # TrashNet dataset downloader & 70/15/15 partitioner
├── train_model.py              # MobileNetV2 Transfer Learning training script
├── evaluate_model.py           # Evaluation script generating genuine metrics & matrix
├── predict.py                  # Single-image inference, uncertainty, & CSV logger
├── generate_docs.py            # Automated DOCX report generator (python-docx)
├── generate_presentation.py    # Automated PPTX presentation generator (python-pptx)
├── requirements.txt            # Python package dependencies
├── run_project.bat             # One-click Windows application launcher
├── train_model.bat             # One-click Windows model training pipeline
├── README.md                   # Comprehensive user guide and documentation
├── .gitignore                  # Git ignore rules for virtualenv and datasets
├── data/
│   ├── raw/                    # Downloaded TrashNet dataset archive
│   └── processed/              # Stratified splits (train/val/test)
├── models/
│   ├── mobilenetv2_recyclable_material.pth   # Trained model weights
│   └── class_labels.json                     # Class-to-index mapping dictionary
├── outputs/
│   ├── graphs/
│   │   ├── training_history.png              # Loss and accuracy curves
│   │   └── confusion_matrix.png              # Unseen test set confusion matrix
│   ├── metrics/
│   │   ├── evaluation_results.json           # Accuracy, F1, Recall, Precision metrics
│   │   ├── class_metrics.csv                 # Per-class performance metrics
│   │   └── dataset_stats.json                # Partition sample counts
│   └── prediction_history.csv                # Session prediction audit log
├── docs/
│   ├── PROJECT_REPORT.md                     # Full academic project report (Markdown)
│   ├── PROJECT_REPORT.docx                   # Formatted Microsoft Word report
│   ├── PROJECT_PRESENTATION.pptx             # 10-slide PowerPoint presentation
│   ├── PRESENTATION_SLIDES.md                # Presentation slide outline
│   ├── VIVA_QUESTIONS_AND_ANSWERS.md         # 25+ viva questions with detailed answers
│   └── DEMO_SCRIPT.md                        # 2-minute project demonstration script
└── tests/
    └── test_pipeline.py                      # Automated pytest unit and integration tests
```

---

## ⚡ Quick Start Guide (One-Click Operation on Windows)

### 1. Launch the Application (Recommended)
Simply double-click:
```bash
run_project.bat
```
This batch script automatically activates the virtual environment, checks for the trained model, and opens the Streamlit application in your default browser at `http://localhost:8501`.

### 2. Manual Command Line Launch
If you prefer running commands in PowerShell or Command Prompt:

```powershell
# 1. Activate the virtual environment
.\venv\Scripts\activate

# 2. Launch the Streamlit application
streamlit run app.py
```

---

## 🛠️ Step-by-Step Training & Evaluation

If you wish to re-train the neural network from scratch or re-evaluate the model:

```powershell
# Step 1: Download & partition the dataset (70% train, 15% val, 15% test)
python download_dataset.py

# Step 2: Fine-tune MobileNetV2 on the dataset
python train_model.py

# Step 3: Evaluate on the unseen test dataset and generate confusion matrix
python evaluate_model.py

# Step 4: Run automated test suite
pytest tests/
```

Or simply double-click `train_model.bat` for an automated end-to-end training pipeline.

---

## 🧪 How to Test an Image

1. Open the application at `http://localhost:8501`.
2. Click **🔍 Material Identification** in the sidebar.
3. Select **📁 Upload Image File** or **📷 Live Camera Capture**.
4. Upload any image (`.jpg`, `.jpeg`, `.png`, `.webp`) or snap a photo of a waste item.
5. View:
   - Predicted material category and confidence score.
   - Confidence breakdown bar chart across all 6 classes.
   - Materials Science chemical & microstructural specifications.
   - Circular recycling guidance and destination bin.
6. Navigate to **⚙️ Automated Sorting Simulation** to view the active diverter chute animation.
7. Navigate to **📜 Prediction History** to download the complete session audit log as a CSV.

---

## 🔬 Connection to Materials Science and Engineering

| Material | Microstructure | Key Physical Properties | Industrial Recycling Mechanism |
| :--- | :--- | :--- | :--- |
| **Plastic** | Long-chain hydrocarbon macromolecules; amorphous/semi-crystalline | Low density (0.9–1.4 g/cm³), insulator, melting 110–260°C | Sink-float density separation, shredding, extrusion; 4–6 cycles |
| **Metal** | Crystalline lattice (FCC for Al, BCC for steel) with metallic electron sea | High conductivity, ductile, high melting point | Magnetic/eddy-current separation, remelting; infinitely recyclable |
| **Paper** | Hydrophilic cellulose microfibrils with hydrogen bonding | High porosity, sensitive to moisture | Hydrapulping, flotation de-inking; 5–7 cycles before fiber shortening |
| **Cardboard** | Arched corrugated medium fluting between linerboards | High compression & burst strength, impact absorption | Repulping, centrifugal glue removal, re-corrugation |
| **Glass** | Amorphous isotropic silica network (\(SiO_2\)) with fluxing cations | Chemically inert, brittle, high hardness | Color optical sorting, crushing into cullet, remelting at 1400°C |
| **Other / Waste** | Multilayer composite laminates & contaminated packaging | Incompatible thermodynamic phases | Refuse-Derived Fuel (RDF) gasification, waste-to-energy |

---

## 📊 Evaluation & Empirical Results

The model was evaluated strictly on the held-out 384 test images with **zero data leakage**:
- **Dataset Partition**:
  - Training: 1,766 images (70%)
  - Validation: 377 images (15%)
  - Test: 384 images (15%)
- **Confusion Matrix**: Saved at `outputs/graphs/confusion_matrix.png`.
- **Training Curves**: Saved at `outputs/graphs/training_history.png`.
- **Per-Class Metrics**: Saved at `outputs/metrics/class_metrics.csv`.

---

## ❓ Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `Model weights not found` | Training has not completed | Run `python train_model.py` or double-click `train_model.bat`. |
| `Camera permission denied` | Browser blocked webcam | Check browser URL bar lock icon and allow camera access. Alternatively, use file upload mode. |
| `ModuleNotFoundError` | Virtual environment inactive | Ensure virtual environment is activated: `.\venv\Scripts\activate`. |
| `Uncertainty warning displayed` | Image confidence < 50% | The item may be a complex composite, poorly illuminated, or non-standard. Try centering under brighter light. |

---

## 🎓 Academic Attribution & License

- **Dataset**: Stanford TrashNet benchmark created by Gary Thung and Mindy Yang (Stanford CS229).
- **Backbone**: MobileNetV2 architecture by Mark Sandler et al. (Google Research, CVPR 2018).
- **Academic Course**: Materials Science and Engineering Laboratory Mini-Project (2026–2027).
