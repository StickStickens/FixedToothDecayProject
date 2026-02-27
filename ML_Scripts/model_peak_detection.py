"""
model_peak_detection.py
-----------------------
Peak detection + XGBoost classifier for Raman spectra.

Fits a pseudo-Voigt profile to the dominant peak in each spectrum,
extracts shape features (position, width, eta, height …), then trains
an XGBClassifier.

Can be run standalone:
    python ML_Scripts/model_peak_detection.py

Or imported and called programmatically (used by run_evaluation.py).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from data_loader import load_all_data, evaluate_model


# =============================================================
# PEAK FITTING
# =============================================================

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def lorentzian(x, a, x0, gamma):
    return a * (gamma ** 2 / ((x - x0) ** 2 + gamma ** 2))


def pseudo_voigt(x, a, x0, sigma, eta, c):
    return eta * lorentzian(x, a, x0, sigma) + (1 - eta) * gaussian(x, a, x0, sigma) + c


def detect_describe_peak(intensities, wavenumbers):
    intensities = np.array(intensities, dtype=float)
    peaks, _ = find_peaks(intensities)
    if len(peaks) == 0:
        return None

    peak_idx = peaks[np.argmax(intensities[peaks])]
    peak_x = wavenumbers[peak_idx]
    peak_y = intensities[peak_idx]

    window = 25
    left = max(0, peak_idx - window)
    right = min(len(wavenumbers), peak_idx + window)
    x_fit = wavenumbers[left:right]
    y_fit = intensities[left:right]

    try:
        popt, _ = curve_fit(
            pseudo_voigt,
            x_fit,
            y_fit,
            p0=[peak_y - np.min(y_fit), peak_x, 3, 0.5, np.min(y_fit)],
            bounds=(
                [0, peak_x - 10, 0, 0, -np.inf],
                [np.inf, peak_x + 10, np.inf, 1, np.inf],
            ),
            maxfev=10000,
        )
    except RuntimeError:
        return None

    a, x0, sigma, eta, c = popt
    fwhm = 2.355 * sigma
    peak_to_background = (a + c) / (c + 1e-9)

    # Derive intensity at the reference wavenumber closest to 959
    wn_arr = np.array(wavenumbers)
    ref_idx = np.argmin(np.abs(wn_arr - 959))
    i_at_959 = intensities[ref_idx]

    return {
        "peak_position": x0,
        "std": sigma,
        "eta": eta,
        "width_fwhm": fwhm,
        "peak_to_background_ratio": peak_to_background,
        "height": a,
        "intensity_at_959": i_at_959,
    }


# =============================================================
# FEATURE EXTRACTION
# =============================================================

def _get_intensity_array(row, intensity_cols, wavenumbers_list):
    return row[intensity_cols].to_numpy(dtype=float)


def peak_detection_transform(df, augmented=False):
    """
    For each row, fit a pseudo-Voigt to the dominant peak and return
    a feature DataFrame.
    """
    intensity_cols = [c for c in df.columns if c.startswith("intensity_at_")]
    wavenumbers = [int(c.split("_")[-1]) for c in intensity_cols]
    wavenumbers_arr = np.array(wavenumbers)

    rows, meta_cols = [], [
        "peak_position", "std", "eta", "width_fwhm",
        "peak_to_background_ratio", "height", "intensity_at_959",
        "Typ_zeba", "ID_zeba", "ID_skanu", "Axis_0", "Axis_1",
    ]
    if augmented:
        meta_cols.append("augmentation_type")


    for _, row in df.iterrows():
        intensities = row[intensity_cols].to_numpy(dtype=float)
        res = detect_describe_peak(intensities, wavenumbers_arr)
        if res is None:
            print(f"Warning: Peak fitting failed for ID_zeba={row['ID_zeba']}, ID_skanu={row['ID_skanu']}, augmented={row.get('augmentation_type', 'N/A')}")
            continue

        entry = [
            res["peak_position"],
            res["std"],
            res["eta"],
            res["width_fwhm"],
            res["peak_to_background_ratio"],
            res["height"],
            res["intensity_at_959"],
            row["Typ_zeba"],
            row["ID_zeba"],
            row["ID_skanu"],
            row["Axis_0"],
            row["Axis_1"],
        ]
        if augmented:
            entry.append(row["augmentation_type"])
        rows.append(entry)

    return pd.DataFrame(rows, columns=meta_cols)


def merge_polarizations(df, augmented=False, polarizations=None):
    """
    Transform each polarisation separately, then inner-join on
    the metadata columns to produce a combined feature table.
    """
    useless = ["ID_zeba", "ID_skanu", "Axis_0", "Axis_1", "Typ_zeba"]
    if augmented:
        useless.append("augmentation_type")

    if not polarizations:
        return peak_detection_transform(df, augmented)

    dfs = [peak_detection_transform(df[df["Polaryzacja"] == p], augmented) for p in polarizations]

    result = dfs[0]
    for i in range(1, len(dfs)):
        result = result.merge(
            dfs[i],
            on=useless,
            how="inner",
            suffixes=("_" + polarizations[i - 1], "_" + polarizations[i]),
        )

    if "vv" in polarizations and "vh" in polarizations:
        # derive depolarisation ratio and anisotropy
        i959_vv = result.get("intensity_at_959_vv", result.get("intensity_at_959"))
        i959_vh = result.get("intensity_at_959_vh", result.get("intensity_at_959"))
        if i959_vv is not None and i959_vh is not None:
            result["rho"] = i959_vh / i959_vv
            result["A"] = (i959_vv - i959_vh) / (i959_vv + 2 * i959_vh)

    return result


# =============================================================
# TRAIN / PREDICT
# =============================================================

def peak_classifier(
    df,
    augmented=False,
    polarizations=None,
    classes=None,
    to_plot_data_42=False,
):
    """
    Train an XGBoost classifier on peak features.

    Parameters
    ----------
    df : DataFrame  – cleaned parquet (intensity_at_XXX columns)
    augmented : bool
    polarizations : list[str] | None  e.g. ['v'] or ['vh', 'vv']
    classes : list[str] | None        filter to these Typ_zeba values
    to_plot_data_42 : bool
        If True, returns df_42 with 'predicted' column for heatmap plotting.
        If False, returns (y_test, y_pred, y_proba) for metric evaluation.
    """
    polarizations = polarizations or ["v"]

    df_copy = df.loc[df["Typ_zeba"].isin(classes)].copy() if classes else df.copy()

    if df_copy["Typ_zeba"].nunique() < len(classes) if classes else 2:
        print("Warning: Not enough classes present for classification. for parameters:", {
            "classes": classes,
            "present_classes": df_copy["Typ_zeba"].unique(),
            "augmented": augmented,
            "polarizations": polarizations,        })
        return (None, None, None) if not to_plot_data_42 else None

    df_features = merge_polarizations(df_copy, augmented, polarizations)
    print("comparison2", df_features["Typ_zeba"].nunique(), len(classes))
    if df_features.empty or df_features["Typ_zeba"].nunique() < len(classes) if classes else 2:
        print("Warning: Not enough classes present for classification. for parameters:", {
            "classes": classes,
            "present_classes": df_copy["Typ_zeba"].unique(),
            "augmented": augmented,
            "polarizations": polarizations,        })
        return (None, None, None) if not to_plot_data_42 else None

    print(f"  Peak feature matrix: {df_features.shape}")

    useless = ["ID_zeba", "ID_skanu", "Axis_0", "Axis_1", "Typ_zeba"]
    if augmented:
        useless.append("augmentation_type")

    df_no42 = df_features[df_features["ID_zeba"] != 42].reset_index(drop=True)
    df_42 = df_features[df_features["ID_zeba"] == 42].reset_index(drop=True)

    encoder = LabelEncoder().fit(df_no42["Typ_zeba"])
    X = df_no42.drop(columns=useless)
    y = encoder.transform(df_no42["Typ_zeba"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    n_classes = len(encoder.classes_)
    model_eval = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=n_classes,
    )
    model_eval.fit(X_train, y_train)
    y_pred = model_eval.predict(X_test)
    y_proba = model_eval.predict_proba(X_test)

    if not to_plot_data_42:
        return y_test, y_pred, y_proba

    # Retrain on all data for tooth-42 prediction
    model_full = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=n_classes,
    )
    model_full.fit(
        np.vstack([X_train, X_test]),
        np.concatenate([y_train, y_test]),
    )

    if df_42.empty:
        print("Warning: No tooth 42 samples found for prediction.")
        return None

    X_42 = df_42.drop(columns=useless)
    disease_class = 1 if encoder.classes_[0] == "Zdrowe" else 0

    df_42 = df_42.copy()
    if not classes or len(classes) == 3:
        preds = model_full.predict(X_42)
        # multi:softprob can return 2D proba array instead of 1D class indices
        if hasattr(preds, "ndim") and preds.ndim == 2:
            preds = np.argmax(preds, axis=1)
        df_42["predicted"] = preds
    else:
        df_42["predicted"] = model_full.predict_proba(X_42)[:, disease_class]

    return df_42


# =============================================================
# STANDALONE
# =============================================================

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    data = load_all_data()
    clean_aug = data["clean_aug"]
    clean_nonaug = data["clean_nonaug"]

    class_sets = [
        ["Chore", "Zdrowe"],
        ["Chore_sztucznie", "Zdrowe"],
        ["Chore", "Zdrowe", "Chore_sztucznie"],
    ]
    polarisation_options = [["v"], ["vh", "vv"], ["vh", "vv", "v"]]

    for classes in class_sets:
        print(f"\n{'='*60}")
        print(f"Classes: {classes}")
        for pols in polarisation_options:
            for aug, df in [(False, clean_nonaug), (True, clean_aug)]:
                label = "aug" if aug else "nonaug"
                y_test, y_pred, y_proba = peak_classifier(
                    df, augmented=aug, polarizations=pols, classes=classes, to_plot_data_42=False
                )
                if y_test is None:
                    print(f"  [{label}] pol={pols}: -------")
                else:
                    auc = evaluate_model(y_test, y_pred, y_proba)
                    print(f"  [{label}] pol={pols}: AUC={auc:.4f}" if isinstance(auc, float) else f"  [{label}] pol={pols}: {auc}")