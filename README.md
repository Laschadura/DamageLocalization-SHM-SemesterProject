# Damage Localization in a Masonry Arch Bridge (ETH Zürich Semester Project)

This repository contains the code and documentation of my **Structural Health Monitoring (SHM)** semester project at **ETH Zürich**.  
The goal of the project was to **detect and localize structural damage** in a full-scale masonry arch bridge using machine learning methods applied to **accelerometer time-series data** and **modal parameters**.

Data originates from the experiment by **Liu et al. (2024)**, where a full-scale bridge was progressively damaged under controlled laboratory conditions.

> **Note:**  
> The raw experimental data cannot be shared publicly.  
> This repository contains **only the code**, documentation, and the project report.

---

## 🚧 Project Motivation

Masonry arch bridges degrade over time due to environmental effects and loading. Manual inspection is slow and subjective.  
The aim of this project was to develop an **ML-based SHM pipeline** capable of:

- Detecting whether the bridge is damaged  
- Localizing the area of damage  
- Comparing classical modal-parameter pipelines vs. raw time-series pipelines  
- Evaluating multiple ML model families (RF, MLP, CNN, attention models)

---

# 📁 Repository Structure
DamageLocalization-SHM-SemesterProject/
│
├── MATLAB/                   # Modal parameter extraction (SSI, FDD, ERA)
│
├── Modal_Parameter_Model/    # ML using modal parameters (RF, MLP, CNN)
│
├── RawDataModel/             # ML using raw accelerometer time series (1D/2D CNNs)
│
└── report/
    └── SP-Report_SHM_Simon_Scandella.pdf


---

# 🧩 Overview of the Methods

### **1. Modal Parameter Extraction (MATLAB)**
Three operational modal analysis methods were automated and applied:

- **SSI-COV** (Stochastic Subspace Identification – Covariance Driven)
- **FDD** (Frequency Domain Decomposition)
- **ERA** (Eigensystem Realization Algorithm)

From these, the following were extracted:

- Natural frequencies  
- Damping ratios  
- Mode shapes  
- Mode shape derivatives (for ERA-based pipelines)  
- MAC-based mode clustering  

These were used as input features for ML-based damage detection and localization.

---

### **2. Zone-based Damage Labels**
Two alternative labeling schemes:

- **10-zone binary labeling**
- **20-zone binary labeling**

Each zone = damaged (1) or undamaged (0).  
Used for multi-output damage classification and localization with classical ML models.

---

### **3. Image-based Damage Masks**
For more detailed localization, damage drawings were transformed into **pixel-level masks**:

- Three bridge views (arch intrados, north spandrel, south spandrel)
- Combined into a **256 × 768** composite mask
- Smoothed using Gaussian filtering for dense supervision

Used for CNN heatmap prediction.

---

### **4. Machine Learning Pipelines**

#### ✔ **Random Forest models (modal parameters)**  
- Multi-output RF models for zone-based damage classification  
- Strongest performer in damage detection  
- Robust even with small datasets

#### ✔ **MLP and 1D CNN models (modal parameters)**  
- Performed well but susceptible to overfitting  
- CNN-SE variant improved feature extraction

#### ✔ **CNN-Attention model (ERA)**  
- Sequence-based encoding of modal parameters  
- Attention mechanism to focus on informative modes  
- Outputs:
  - **Binary damage detection**
  - **256×768 damage heatmap**

#### ✔ **1D–2D CNN model (raw accelerometer time series)**  
- Encodes time-series → decodes into 2D damage localization heatmap  
- Inspired by:
  - Dang et al. (2021)  
  - Lin et al. (2017)

---

# 🧪 Key Results

### **Damage Detection**
- **Random Forest (FDD dataset):**
  - **Accuracy:** 92.11%
  - **Recall:** 90.25%
  - **F1:** 94.49%
- NNs achieve high recall but weaker precision due to limited data.

### **Damage Localization**
- **Zone-based localization:** RF best overall  
- **Image-based CNN heatmaps:** moderate Dice scores  
  - Challenging due to label sparsity and limited severe-damage cases  
- **Time-series CNN:** good localization *after* adding high-damage augmentations  
  - But this introduces data leakage → reduced generalization reliability

---

# 🧠 Key Insights

- Modal-parameter ML is effective for **damage detection**, less so for precise localization.
- CNN localization models require **many more high-damage samples** than available.
- Chronological splitting avoids data leakage but makes the learning task harder.
- Augmenting high-damage scenarios helps, but biases the evaluation.
- RF remains extremely strong when data is small and heterogeneous.

---

# 📌 Limitations

- Only 23 usable damage scenarios → deep learning models overfit quickly  
- Damage masks are sparse → difficult for pixel-level supervision  
- Modal parameter extraction contains noise and inconsistent mode tracking  
- High-damage cases appear only at the end of the test sequence  

---

# 🚀 Future Work

- Collect more high-damage and intermediate-damage scenarios  
- Use simulation (e.g., FEM + modal synthesis) to generate synthetic data  
- Apply **unsupervised** methods (autoencoders, anomaly detection)  
- Explore Bayesian or probabilistic ML (e.g., simulation-based inference)  
- Better mode-tracking pipelines for robust modal identification  

---

# 📖 Full Report

The detailed methodology, visualizations, results, and discussion can be found in:

👉 `report/SP-Report_SHM_Simon_Scandella.pdf`

---

# 👨‍💻 Author

**Simon Scandella**  
MSc Mechanical Engineering – ETH Zürich  
Focus: Energy Technologies, Machine Learning, Structural Health Monitoring  
Email: simon.scandellas@gmail.com  
GitHub: [@Laschadura](https://github.com/Laschadura)

---

# ⭐ If you found this project interesting  
Feel free to star the repo or reach out — I’m always open to discuss SHM, ML, and simulation projects!

