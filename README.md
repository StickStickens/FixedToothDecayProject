# FixedToothDecayProject
new version

data and drafts
https://drive.google.com/drive/folders/1cFnvtjBtaS-p99hQgCMu7KeNfEVWM2Gp

discussion and notes:
- https://docs.google.com/document/d/1XsOTuKXdImdlv5nC_OHXEjU2BeNwqM5kngFqdzl1OXk/edit?tab=t.0

# Description
Tooth Decay Detection from Raman Spectroscopy

This repository contains a machine learning pipeline for automatic detection of dental enamel decay using Raman spectroscopy. The goal is to enable non-invasive and objective diagnosis without manual interpretation of spectral data.

The dataset includes 10,249 Raman spectra from 43 tooth regions, with labels for healthy, artificially decayed, and naturally decayed enamel. Due to strong class imbalance, realistic data augmentation (noise, baseline drift, peak shifts, intensity scaling, and spectral warping) was applied to decayed samples.

Three modeling approaches were evaluated:

MiniRocket – fast feature extraction for spectral sequences with a Ridge classifier

XGBoost peak classifier – interpretable model using manually extracted spectral features from the ν₁ PO₄³⁻ band (900–990 cm⁻¹)

Deep Sets – neural architecture that processes spectra as sets of (wavenumber, intensity) pairs without interpolation

Models were evaluated using a tooth-level train/test split to avoid data leakage. The best models achieved AUC values close to 1.0, showing strong diagnostic performance.

Interpretability analysis (SHAP and Deep Sets mapping) confirmed that predictions rely on physically meaningful biomarkers such as peak position, FWHM, and Raman intensity.

The results demonstrate the potential of combining Raman spectroscopy and machine learning for automated early dental diagnostics.

<img width="609" height="523" alt="image" src="https://github.com/user-attachments/assets/e6f1cb07-d823-4952-8de4-e087c4c4abe0" />

