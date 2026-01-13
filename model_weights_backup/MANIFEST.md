# Model Weights Manifest

This directory contains backup copies of the best model checkpoints for the ECG Agent tasks.

## 1. Classification
*   **Task**: ECG Signal Classification (CPSC2018)
*   **Original Path**: `/data1/tangzhijiang/ECG/LTM/LECGM/checkpoints/PSSM_V015/PSSM_V015_d16_cpsc_20181.0_finerate0.5_undersample_lr1e-05_Classification/best_model.pth`
*   **Backup File**: `classification_best_model.pth`
*   **Configuration**: `d_model=16`, `num_classes=7`

## 2. Detection
*   **Task**: ECG Event Detection (BA-LABOUR)
*   **Original Path**: `/data1/tangzhijiang/ECG/LTM/LECGM/checkpoints/PSSM_V015/PSSM_V015_d16_BA-LABOUR1.0_finerate0.5_0_lr5e-05_frozen0_Detection/best_model.pth`
*   **Backup File**: `detection_best_model.pth`
*   **Configuration**: `d_model=16`

## 3. Forecasting
*   **Task**: ECG Signal Forecasting (PTB)
*   **Original Path**: `/data1/tangzhijiang/ECG/LTM/LECGM/checkpoints/PSSM_V015/PSSM_V015_ptb1.0_predict100_finerate0.5_0_lr1e-05_Forecast/best_model.pth`
*   **Backup File**: `forecast_best_model.pth`
*   **Configuration**: `d_model=16`

## 4. Generation
*   **Task**: ECG Signal Generation/Separation (MITDB)
*   **Original Path**: `/data1/tangzhijiang/ECG/LTM/LECGM/checkpoints/PSSM_V015/PSSM_V015_d16_MITDB1.0_finerate0.5_0_lr5e-05_frozen0_Generation/best_model.pth`
*   **Backup File**: `generation_best_model.pth`
*   **Configuration**: `d_model=16`
