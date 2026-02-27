"""
model_minirocket.py
-------------------
MiniRocket + Ridge classifier for Raman spectra.

Treats each cleaned spectrum (intensity_at_XXX columns) as a univariate
time series and uses MiniRocket to generate ~10 000 random-convolutional
features, then fits a Ridge classifier with cross-validated alpha.

Can be run standalone:
    python ML_Scripts/model_minirocket.py

Or imported and called programmatically (used by run_evaluation.py).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import softmax
from sktime.transformations.panel.rocket import MiniRocket

from data_loader import load_all_data, evaluate_model


# =============================================================
# RIDGE WITH PROBA
# =============================================================

class RidgeClassifierWithProba(RidgeClassifierCV):
    def predict_proba(self, X):
        d = self.decision_function(X)
        if d.ndim == 1:
            d = np.c_[-d, d]
        return softmax(d)


# =============================================================
# DATA PREPARATION
# =============================================================

def _get_intensity_cols(df):
    return [c for c in df.columns if c.startswith("intensity_at_")]


def prepare_dataset(df, augmented=False, polarizations=None):
    """
    Filter by polarisation(s), merge if multiple, and return
    (train_df, test_df) each with an 'intensity_vector' column.

    When multiple polarisations are requested the intensity vectors from
    each are concatenated, effectively giving the classifier all channels.
    """
    useless = ["Typ_zeba", "ID_zeba", "ID_skanu", "Is_single_place",
               "Axis_0", "Axis_1", "time"]
    if augmented:
        useless.append("augmentation_type")

    polarizations = polarizations or ["v"]
    intensity_cols_base = _get_intensity_cols(df)

    if len(polarizations) == 1:
        pol_df = df[df["Polaryzacja"] == polarizations[0]].copy()
        # Each cell must be a pd.Series — the nested format sktime expects
        pol_df["intensity_vector"] = [
            pd.Series(row, dtype=float)
            for row in pol_df[intensity_cols_base].to_numpy(dtype=float)
        ]
    else:
        # Merge polarisations side-by-side, then concatenate intensity vectors
        merged = None
        for pol in polarizations:
            sub = df[df["Polaryzacja"] == pol].copy()
            rename = {c: f"{c}_{pol}" for c in intensity_cols_base}
            sub = sub.rename(columns=rename)
            key_cols = [c for c in useless if c in sub.columns]
            if merged is None:
                merged = sub
            else:
                merged = merged.merge(
                    sub, on=key_cols, how="inner",
                    suffixes=("", f"_{pol}")
                )

        all_int_cols = [f"{c}_{p}" for p in polarizations for c in intensity_cols_base]
        all_int_cols = [c for c in all_int_cols if c in merged.columns]
        merged["intensity_vector"] = [
            pd.Series(row, dtype=float)
            for row in merged[all_int_cols].to_numpy(dtype=float)
        ]
        pol_df = merged

    # Drop tooth 42 from train/test split
    pol_no42 = pol_df[pol_df["ID_zeba"] != 42].reset_index(drop=True)

    train_df, test_df = train_test_split(pol_no42, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df, pol_df


# =============================================================
# TRAIN / PREDICT
# =============================================================

def predict_with_minirocket(
    df,
    augmented=False,
    polarizations=None,
    classes=None,
    to_plot_data_42=False,
):
    """
    Train MiniRocket + Ridge on df, evaluate, and optionally return
    tooth-42 predictions for heatmap plotting.

    Parameters
    ----------
    df : DataFrame  – cleaned parquet (intensity_at_XXX columns)
    augmented : bool
    polarizations : list[str] | None
    classes : list[str] | None  – filter to these Typ_zeba values
    to_plot_data_42 : bool
        If True, returns df_42 with 'predicted' column.
        If False, returns (y_test, y_pred, y_proba).
    """
    polarizations = polarizations or ["v"]

    df_copy = df[df["Typ_zeba"].isin(classes)].copy() if classes else df.copy()
    print("classes", classes)
    print("comparison", df_copy["Typ_zeba"].nunique(), len(classes))
    if df_copy["Typ_zeba"].nunique() < len(classes) if classes else 2:
        return (None, None, None) if not to_plot_data_42 else None

    train_df, test_df, full_pol_df = prepare_dataset(df_copy, augmented, polarizations)

    print("comparison2", train_df["Typ_zeba"].nunique(), len(classes))
    if train_df["Typ_zeba"].nunique() < len(classes) if classes else 2 or test_df["Typ_zeba"].nunique() < len(classes) if classes else 2:
        return (None, None, None) if not to_plot_data_42 else None

    encoder = LabelEncoder().fit(train_df["Typ_zeba"].unique())

    X_train = pd.DataFrame(train_df["intensity_vector"])
    y_train = encoder.transform(train_df["Typ_zeba"])
    X_test = pd.DataFrame(test_df["intensity_vector"])
    y_test = encoder.transform(test_df["Typ_zeba"])

    print(f"  MiniRocket fitting on {X_train.shape[0]} samples …")
    rocket = MiniRocket()
    rocket.fit(X_train)

    scaler = StandardScaler(with_mean=False)
    classifier = RidgeClassifierWithProba(
        alphas=np.logspace(-3, 3, 10), class_weight="balanced"
    )

    X_train_t = scaler.fit_transform(rocket.transform(X_train))
    classifier.fit(X_train_t, y_train)

    X_test_t = scaler.transform(rocket.transform(X_test))
    y_pred = classifier.predict(X_test_t)
    y_proba = classifier.predict_proba(X_test_t)

    if not to_plot_data_42:
        return y_test, y_pred, y_proba

    # Get tooth-42 rows (from the full polarisation-filtered frame)
    _, _, full_for_42 = prepare_dataset(df, augmented, polarizations)
    df_42 = full_for_42[full_for_42["ID_zeba"] == 42].reset_index(drop=True)

    if df_42.empty:
        return None

    X_42 = pd.DataFrame(df_42["intensity_vector"])
    X_42_t = scaler.transform(rocket.transform(X_42))

    disease_class = 1 if encoder.classes_[0] == "Zdrowe" else 0
    df_42 = df_42.copy()

    if not classes or len(classes) == 3:
        df_42["predicted"] = classifier.predict(X_42_t)
    else:
        df_42["predicted"] = classifier.predict_proba(X_42_t)[:, disease_class]

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
                y_test, y_pred, y_proba = predict_with_minirocket(
                    df, augmented=aug, polarizations=pols,
                    classes=classes, to_plot_data_42=False
                )
                if y_test is None:
                    print(f"  [{label}] pol={pols}: -------")
                else:
                    auc = evaluate_model(y_test, y_pred, y_proba)
                    print(f"  [{label}] pol={pols}: AUC={auc:.4f}" if isinstance(auc, float) else f"  [{label}] pol={pols}: {auc}")