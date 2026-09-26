# ==============================================================================
# PREDICTIVE VEHICLE MAINTENANCE WITH ROAD ANALYSIS USING DEEP LEARNING
# Standalone Execution Pipeline
# ==============================================================================

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn & Imbalanced-Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from imblearn.over_sampling import SMOTE

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Explainable AI
import shap

# Configuration & Directories
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def find_dataset():
    candidates = [
        os.path.join(BASE_DIR, "data", "raw", "logistics_predictive_maintenanceV2.csv"),
        os.path.join(BASE_DIR, "..", "data", "raw", "logistics_predictive_maintenanceV2.csv"),
        os.path.join(BASE_DIR, "logistics_predictive_maintenanceV2.csv"),
        "data/raw/logistics_predictive_maintenanceV2.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Could not find logistics_predictive_maintenanceV2.csv in data/raw/")

def build_dnn_model(input_dim, name="Predictive_Maintenance_DNN"):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu', name="Dense_128"),
        BatchNormalization(name="BatchNorm_1"),
        Dropout(0.30, name="Dropout_1"),
        
        Dense(64, activation='relu', name="Dense_64"),
        BatchNormalization(name="BatchNorm_2"),
        Dropout(0.30, name="Dropout_2"),
        
        Dense(32, activation='relu', name="Dense_32"),
        Dropout(0.20, name="Dropout_3"),
        
        Dense(1, activation='sigmoid', name="Output_Sigmoid")
    ], name=name)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

def main():
    print("=" * 80)
    print("PREDICTIVE VEHICLE MAINTENANCE WITH ROAD ANALYSIS (DEEP LEARNING PIPELINE)")
    print("=" * 80)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPUs Detected:      {tf.config.list_physical_devices('GPU')}")

    # 1. Load Data
    data_path = find_dataset()
    print(f"\n[Step 1/8] Loading dataset from: {data_path}")
    df_raw = pd.read_csv(data_path)
    print(f"Raw shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")

    # 2. Leakage Removal
    print("\n[Step 2/8] Removing 25 data leakage and synthetic outcome variables...")
    leakage_cols = [
        'Vehicle_ID', 'Last_Maintenance_Date', 'Predictive_Score', 'PCR', 'UIR',
        'TPI', 'MBF', 'ADS', 'OHI', 'CMES', 'UER', 'Maintenance_Cost',
        'Downtime_Maintenance', 'Maintenance_Type', 'Maintenance_Severity',
        'Maintenance_Severity_ID', 'FL_Client_ID', 'Partition_Type',
        'Local_Epochs_Per_Round', 'Communication_Rounds', 'Data_Split',
        'Pre_Event_Record', 'Historical_Maintenance_Cost',
        'Days_Since_Last_Maintenance', 'Impact_on_Efficiency', 'is_noisy',
        'Communication_Interface', 'Telematics_Gateway', 'Edge_Device_Class',
        'Brake_Condition'
    ]
    cols_to_drop = [c for c in leakage_cols if c in df_raw.columns]
    df_clean = df_raw.drop(columns=cols_to_drop).copy()
    print(f"Purged {len(cols_to_drop)} columns. Remaining: {len(df_clean.columns)}")

    # 3. Base Predictor Alignment
    BASE_FEATURES = [
        'Make_and_Model', 'Vehicle_Type', 'Year_of_Manufacture', 'Road_Conditions',
        'Weather_Conditions', 'Route_Info', 'Usage_Hours', 'Load_Capacity',
        'Actual_Load', 'Engine_Temperature', 'Fuel_Consumption', 'Battery_Status',
        'Oil_Quality', 'Vibration_Levels', 'Tire_Pressure', 'Failure_History',
        'Anomalies_Detected', 'Diagnostic_Trouble_Code_Count',
        'CAN_Message_Rate_Hz', 'Sensor_Packet_Loss_Rate'
    ]
    TARGET = 'Maintenance_Required'
    df_selected = df_clean[BASE_FEATURES + [TARGET]].copy()

    # 4. Feature Engineering & Categorical Encoding
    print("\n[Step 3/8] Performing categorical encoding & road-context feature engineering...")
    cat_cols = ['Make_and_Model', 'Vehicle_Type', 'Road_Conditions', 'Weather_Conditions', 'Route_Info']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_selected[col] = le.fit_transform(df_selected[col].astype(str))
        encoders[col] = le

    df_selected['Load_Utilization'] = df_selected['Actual_Load'] / np.maximum(df_selected['Load_Capacity'], 1.0)
    df_selected['Road_Load_Interaction'] = df_selected['Road_Conditions'] * df_selected['Load_Utilization']
    df_selected['Road_Usage_Interaction'] = df_selected['Road_Conditions'] * df_selected['Usage_Hours']

    ENGINEERED_ROAD_FEATURES = ['Load_Utilization', 'Road_Load_Interaction', 'Road_Usage_Interaction']
    ALL_FEATURES = BASE_FEATURES + ENGINEERED_ROAD_FEATURES
    print(f"Constructed {len(ENGINEERED_ROAD_FEATURES)} road interaction features. Total features: {len(ALL_FEATURES)}")

    # 5. Stratified Split (80/20)
    print("\n[Step 4/8] Splitting data into 80% train and 20% untouched held-out test...")
    X = df_selected[ALL_FEATURES]
    y = df_selected[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")

    # 6. SMOTE Balancing on Training Set ONLY
    print("\n[Step 5/8] Applying SMOTE oversampling exclusively on training data...")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Balanced train records: {len(X_train_smote):,} (50/50 balance achieved)")

    # 7. Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)

    # 8. Train Road-Aware DNN Model
    print("\n[Step 6/8] Building and training Deep Neural Network (MLP)...")
    model = build_dnn_model(input_dim=X_train_scaled.shape[1], name="Road_Aware_DNN")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=os.path.join(MODELS_DIR, "best_checkpoint.keras"), monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(
        X_train_scaled, y_train_smote,
        epochs=35,
        batch_size=256,
        validation_split=0.20,
        callbacks=callbacks,
        verbose=1
    )

    # 9. Evaluate on Held-out Test Set
    print("\n[Step 7/8] Evaluating model on untouched held-out test set (N=50,000)...")
    y_pred_probs = model.predict(X_test_scaled, batch_size=512).ravel()
    y_pred_binary = (y_pred_probs >= 0.50).astype(int)

    acc = accuracy_score(y_test, y_pred_binary)
    prec = precision_score(y_test, y_pred_binary)
    rec = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    auc = roc_auc_score(y_test, y_pred_probs)

    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET METRICS (THRESHOLD = 0.50)")
    print("=" * 60)
    print(f"Accuracy:   {acc * 100:.2f}%")
    print(f"Precision:  {prec * 100:.2f}%")
    print(f"Recall:     {rec * 100:.2f}%")
    print(f"F1-Score:   {f1 * 100:.2f}%")
    print(f"ROC-AUC:    {auc * 100:.2f}%")
    print("=" * 60)
    print("\nClassification Report:\n", classification_report(y_test, y_pred_binary))

    # Save Evaluation Visualizations
    cm = confusion_matrix(y_test, y_pred_binary)
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax1,
                xticklabels=['No Maint (0)', 'Maint Req (1)'],
                yticklabels=['No Maint (0)', 'Maint Req (1)'])
    ax1.set_title('Test Set Confusion Matrix', fontweight='bold')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')

    ax2.plot(fpr, tpr, color='#2563eb', lw=2.5, label=f'DNN (AUC = {auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', lw=1.5)
    ax2.set_title('Receiver Operating Characteristic (ROC)', fontweight='bold')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend()

    eval_plot_path = os.path.join(OUTPUTS_DIR, "evaluation_metrics.png")
    plt.tight_layout()
    plt.savefig(eval_plot_path, dpi=300)
    plt.close()
    print(f"Evaluation curves saved to: {eval_plot_path}")

    # 10. Controlled Road-Context Ablation Study
    print("\n[Step 8/8] Conducting controlled Road-Context Ablation Experiment...")
    NON_ROAD_FEATURES = [
        'Make_and_Model', 'Vehicle_Type', 'Year_of_Manufacture', 'Usage_Hours',
        'Load_Capacity', 'Actual_Load', 'Engine_Temperature', 'Fuel_Consumption',
        'Battery_Status', 'Oil_Quality', 'Vibration_Levels', 'Tire_Pressure',
        'Failure_History', 'Anomalies_Detected', 'Diagnostic_Trouble_Code_Count',
        'CAN_Message_Rate_Hz', 'Sensor_Packet_Loss_Rate'
    ]
    X_train_nr = X_train[NON_ROAD_FEATURES]
    X_test_nr = X_test[NON_ROAD_FEATURES]

    smote_nr = SMOTE(random_state=RANDOM_STATE)
    X_train_nr_s, y_train_nr_s = smote_nr.fit_resample(X_train_nr, y_train)

    scaler_nr = StandardScaler()
    X_train_nr_scaled = scaler_nr.fit_transform(X_train_nr_s)
    X_test_nr_scaled = scaler_nr.transform(X_test_nr)

    model_baseline = build_dnn_model(input_dim=len(NON_ROAD_FEATURES), name="Baseline_NonRoad_DNN")
    model_baseline.fit(
        X_train_nr_scaled, y_train_nr_s,
        epochs=35, batch_size=256, validation_split=0.20,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)],
        verbose=0
    )

    y_pred_probs_nr = model_baseline.predict(X_test_nr_scaled, batch_size=512).ravel()
    y_pred_bin_nr = (y_pred_probs_nr >= 0.50).astype(int)

    acc_nr = accuracy_score(y_test, y_pred_bin_nr)
    prec_nr = precision_score(y_test, y_pred_bin_nr)
    rec_nr = recall_score(y_test, y_pred_bin_nr)
    f1_nr = f1_score(y_test, y_pred_bin_nr)
    auc_nr = roc_auc_score(y_test, y_pred_probs_nr)

    ablation_df = pd.DataFrame({
        'Experiment': ['Baseline DNN (Without Road Context)', 'Road-Aware DNN (With Road Context)'],
        'Features': [len(NON_ROAD_FEATURES), len(ALL_FEATURES)],
        'Accuracy (%)': [acc_nr * 100, acc * 100],
        'Precision (%)': [prec_nr * 100, prec * 100],
        'Recall (%)': [rec_nr * 100, rec * 100],
        'F1-Score (%)': [f1_nr * 100, f1 * 100],
        'ROC-AUC (%)': [auc_nr * 100, auc * 100]
    })
    print("\n" + "=" * 75)
    print("ROAD-CONTEXT ABLATION COMPARISON")
    print("=" * 75)
    print(ablation_df.round(2).to_string(index=False))

    # Save Ablation Chart
    metrics = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'ROC-AUC (%)']
    m_nr = [acc_nr * 100, prec_nr * 100, rec_nr * 100, f1_nr * 100, auc_nr * 100]
    m_r = [acc * 100, prec * 100, rec * 100, f1 * 100, auc * 100]

    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, m_nr, width, label='Without Road Context', color='#94a3b8')
    plt.bar(x + width/2, m_r, width, label='Road-Aware DNN', color='#2563eb')
    plt.ylabel('Score (%)')
    plt.title('Ablation Study: Contribution of Road-Contextual Features', fontweight='bold')
    plt.xticks(x, metrics)
    plt.ylim(80, 100)
    plt.legend()
    ablation_plot_path = os.path.join(OUTPUTS_DIR, "ablation_comparison.png")
    plt.tight_layout()
    plt.savefig(ablation_plot_path, dpi=300)
    plt.close()

    # 11. Save Models & Scalers
    keras_path = os.path.join(MODELS_DIR, "predictive_vehicle_maintenance_dnn.keras")
    model.save(keras_path)

    scaler_path = os.path.join(MODELS_DIR, "telemetry_standard_scaler.pkl")
    joblib.dump(scaler, scaler_path)

    meta_path = os.path.join(MODELS_DIR, "model_feature_metadata.json")
    metadata = {
        "base_features": BASE_FEATURES,
        "engineered_road_features": ENGINEERED_ROAD_FEATURES,
        "all_features": ALL_FEATURES,
        "target": TARGET,
        "classification_threshold": 0.50,
        "final_metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4)
        }
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 80)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"- Model saved to:    {keras_path}")
    print(f"- Scaler saved to:   {scaler_path}")
    print(f"- Metadata saved to: {meta_path}")
    print(f"- Visualizations:    {OUTPUTS_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()
