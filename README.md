# Predictive Vehicle Maintenance with Road Analysis Using Deep Learning

**Final-Year Undergraduate Engineering Research Project**  
*A standalone, deep-learning-based predictive maintenance framework for commercial logistics fleets integrating vehicle telemetry and road-context analysis.*

---

## 📌 Project Overview

This project implements an end-to-end Deep Learning pipeline using a Multi-Layer Perceptron (MLP) with Batch Normalization, Dropout regularization, and Early Stopping. It predicts whether a commercial logistics vehicle requires maintenance (`Maintenance_Required = 1` vs `0`) by synthesizing:

1. **Mechanical & Powertrain Telemetry**: Engine temperature, vibration levels, oil quality, fuel consumption.
2. **Electrical & Communication Telemetry**: Battery status, CAN message rate, sensor packet loss rate, diagnostic trouble code (DTC) counts.
3. **Operational Specifications**: Vehicle class, rated capacity, actual payload load, engine operating hours.
4. **Environmental & Road Context**: Road surface condition, ambient weather, and logistics route classification.

---

## 📁 Project Directory Structure

```
Predictive-Vehicle-Maintenance-Deep-Learning/
│
├── data/
│   └── raw/
│       └── logistics_predictive_maintenanceV2.csv    # 250,000-record dataset (~183 MB)
│
├── models/                                           # Serialized trained artifacts
│   ├── predictive_vehicle_maintenance_dnn.keras     # Final trained Keras deep learning model
│   ├── telemetry_standard_scaler.pkl                 # Scaler fitted on training data
│   └── model_feature_metadata.json                   # Feature names and evaluation metrics
│
├── notebooks/                                        # Google Colab / Jupyter notebook
│   └── Predictive_Vehicle_Maintenance_Deep_Learning.ipynb
│
├── outputs/                                          # Generated evaluation plots and figures
│   ├── evaluation_metrics.png
│   └── ablation_comparison.png
│
├── src/                                              # Reusable modular source code
│   └── train_dl_model.py
│
├── Predictive_Vehicle_Maintenance_Deep_Learning.ipynb # Top-level copy for quick access
├── train_dl_model.py                                 # Single-command executable training script
├── requirements.txt                                  # Environment dependencies
├── .gitignore                                        # Standard Git ignore rules
└── README.md                                         # Project documentation
```

---

## 🔬 Key Engineering Pipeline Steps

1. **Data Leakage Removal**: Systematically purges 25 post-event, synthetic indices (`PCR`, `ADS`, `UIR`, etc.), and client metadata columns.
2. **20 Verified Base Predictors**: Retains strictly realistic operational variables.
3. **Road-Context Feature Engineering**:
   - $\text{Load\_Utilization} = \frac{\text{Actual\_Load}}{\text{Load\_Capacity}}$
   - $\text{Road\_Load\_Interaction} = \text{Road\_Conditions} \times \text{Load\_Utilization}$
   - $\text{Road\_Usage\_Interaction} = \text{Road\_Conditions} \times \text{Usage\_Hours}$
4. **Stratified 80/20 Train-Test Split**: Completely holds out 50,000 test records (`random_state=42`).
5. **Class Imbalance Mitigation**: SMOTE applied **strictly to the training partition** (test partition is never oversampled).
6. **Feature Scaling**: `StandardScaler` fitted **only** on the training partition.
7. **Deep Neural Network (DNN)**:
   - `Dense(128, ReLU)` $\to$ `BatchNormalization` $\to$ `Dropout(0.30)`
   - `Dense(64, ReLU)` $\to$ `BatchNormalization` $\to$ `Dropout(0.30)`
   - `Dense(32, ReLU)` $\to$ `Dropout(0.20)`
   - `Dense(1, Sigmoid)` (Decision threshold: $P \ge 0.50$)
8. **Explainable AI (SHAP)**: Feature attribution ranking and single-vehicle risk explanations using `shap.KernelExplainer`.
9. **Controlled Road-Context Ablation Study**: Side-by-side comparison isolating the quantitative impact of road context.

---

## 🚀 How to Run

### Method 1: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com).
2. Click **File** $\to$ **Upload Notebook** and choose `Predictive_Vehicle_Maintenance_Deep_Learning.ipynb`.
3. (Optional) Set runtime to **GPU** (`Runtime` $\to$ `Change runtime type` $\to$ `T4 GPU`).
4. Click **Runtime** $\to$ **Run all**.

### Method 2: Jupyter Notebook / VS Code (Locally)
1. Navigate to the project root:
   ```bash
   cd E:\Predictive-Vehicle-Maintenance-Deep-Learning
   ```
2. Activate your virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook Predictive_Vehicle_Maintenance_Deep_Learning.ipynb
   ```
4. Click **Run All**.

### Method 3: Command Line (One-Command Script)
To run the complete pipeline directly from your terminal:
```bash
python train_dl_model.py
```
This trains the model, computes metrics on the held-out test set, runs the ablation experiment, and exports plots to `outputs/` and models to `models/`.

---

## 📊 Summary of Experimental Results (Test Set N = 50,000)

| Metric | Non-Road Baseline DNN | Road-Aware DNN (Proposed) | Improvement ($\Delta$) |
|---|:---:|:---:|:---:|
| **Accuracy** | 89.2% | **93.8%** | **+4.6%** |
| **Precision** | 87.5% | **91.4%** | **+3.9%** |
| **Recall** | 83.1% | **88.6%** | **+5.5%** |
| **F1-Score** | 85.2% | **89.9%** | **+4.7%** |
| **ROC-AUC** | 94.1% | **97.2%** | **+3.1%** |

---

## ⚖️ Research Limitations & Viva Defense Notes
- **Historical vs. Live Data**: Telemetry is evaluated on historical logs; physical CAN-bus transmission latency was not measured in real-time.
- **Explainability Boundary**: SHAP indicates mathematical model attribution; it does not constitute physical proof of mechanical failure.
