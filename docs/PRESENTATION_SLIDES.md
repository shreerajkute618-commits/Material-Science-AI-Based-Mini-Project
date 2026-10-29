# 10-SLIDE PRESENTATION OUTLINE
## AI-Based Identification and Sorting of Recyclable Materials
### Materials Science and Engineering Department

---

### **Slide 1: Title Slide**
- **Title:** AI-Based Identification and Sorting of Recyclable Materials
- **Subtitle:** Computer Vision and Deep Learning for Circular Materials Engineering
- **Student Name:** `[STUDENT NAME - EDITABLE]` (Roll No: `[ROLL NUMBER]`)
- **Department:** Materials Science and Engineering
- **Institution:** `[COLLEGE NAME - EDITABLE]`
- **Academic Year:** 2026 – 2027

---

### **Slide 2: Background & Problem Statement**
- Global municipal solid waste generation is escalating rapidly (>2 billion tonnes/year).
- Traditional recycling facilities (MRFs) rely heavily on manual hand-picking.
- Major challenges:
  - High contamination rates in recycled scrap bales (>20%).
  - Severe worker health hazards (sharp metals, bioaerosols, toxic chemicals).
  - High operational costs and slow sorting speeds.
- Need: Automated, sensor-driven, high-speed material segregation.

---

### **Slide 3: Project Objectives**
- Develop an AI vision pipeline classifying materials into 6 circular economy classes:
  - Plastic, Metal, Paper, Cardboard, Glass, Other/Refuse.
- Bridge computer vision predictions with fundamental Materials Science properties.
- Guarantee zero data leakage using an empirical 70/15/15 train/val/test protocol.
- Integrate uncertainty thresholds to safeguard against erroneous misclassifications.
- Build an interactive web UI with real-time sorting chute simulation and CSV logs.

---

### **Slide 4: Materials Science Integration**
- **Plastics:** Polymer macromolecular structures, resin degradation limits, density sorting.
- **Metals:** Crystalline metallic lattices (FCC/BCC), high conductivity, infinite recyclability without property loss.
- **Paper & Cardboard:** Hydrophilic cellulose fiber webs, hydrogen bonding, fiber shortening limits.
- **Glass:** Amorphous silicate (\(SiO_2\)) networks, cullet melting thermodynamics, infinite recyclability.
- **Other/Composites:** Non-separable multi-material packaging diverted to Refuse-Derived Fuel.

---

### **Slide 5: System Architecture & Data Flow**
- Input: Image Upload or Live Webcam feed.
- Preprocessing: Resize to 224×224 pixels, RGB conversion, ImageNet normalization.
- Feature Extractor: MobileNetV2 Deep Convolutional Neural Network.
- Classifier Head: Global Average Pooling + Dropout + Linear Layers + Softmax.
- Decision Layer: Uncertainty Guardrail (<50% confidence warning).
- Actuation: Sorting Chute Digital Twin & CSV Audit Log.

---

### **Slide 6: Deep Learning Methodology & MobileNetV2**
- Transfer Learning from ImageNet dataset.
- Why MobileNetV2?
  - Depthwise Separable Convolutions (splits 3D spatial & channel filtering).
  - Inverted Residual Blocks with linear bottlenecks.
  - Ultra-lightweight: ~3.4 million parameters (compared to 138M in VGG16).
  - High throughput suitable for industrial edge computing and sorting belts.

---

### **Slide 7: Dataset & Rigorous Partitioning**
- Benchmark: Stanford TrashNet Dataset (2,527 images across 6 classes).
- Partitioning Protocol:
  - Training Set: 1,766 images (70%) with data augmentation.
  - Validation Set: 377 images (15%) for hyperparameter tuning.
  - Test Set: 384 images (15%) strictly held out for final evaluation.
- Prevents data leakage between experimental sets.

---

### **Slide 8: Experimental Evaluation & Results**
- Evaluation strictly on held-out test dataset:
  - Multi-class Confusion Matrix analysis.
  - Overall accuracy, Precision, Recall, and F1-Scores.
  - Stable cross-entropy loss convergence across training epochs.
- Cardboard, Glass, and Metal achieve highest precision due to distinctive structural textures and reflectivity.
- Flexible plastics and paper occasionally overlap, accurately caught by uncertainty safety filters.

---

### **Slide 9: Automated Sorting Simulation & UI Features**
- Streamlit Web Application:
  - Instant classification from photo or webcam.
  - Probability distribution bar chart across all 6 classes.
  - Material science properties card (microstructure, density, melting point).
  - Industrial conveyor belt digital twin highlighting active diverter chute.
  - Real-time session history log with one-click CSV export.

---

### **Slide 10: Conclusion & Future Engineering Scope**
- **Conclusion:** Successfully validated an automated, materials-aware recyclable sorting system combining computer vision with sustainable circular engineering.
- **Future Scope:**
  - Hyperspectral Short-Wave Infrared (SWIR) sensors to distinguish specific plastic resins (PET vs. PP).
  - 3D Depth camera integration for robotic gripper pose estimation.
  - Hardware deployment on Raspberry Pi with pneumatic air-jet solenoid valves.
