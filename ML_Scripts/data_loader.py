"""
data_loader.py
--------------
Shared data loading and preparation utilities for all model scripts.

Loads the new-format parquet files and exposes the same logical dataframes
that table_creator_v3.ipynb used, without needing the old split-by-polarisation
files. The 'Polaryzacja' column is used for filtering at query time.

Files required in Data/:
  - scans_clean_augmented.parquet     (cleaned, has augmentation_type column)
  - scans_clean_nonaugmented.parquet  (cleaned, no augmentation_type column)
  - scans_augmented.parquet           (raw, has Wavenumbers/Intensities columns)
  - scans_nonaugmented.parquet        (raw, has Wavenumbers/Intensities columns)
  - detected_grad_42.csv              (axis remapping for tooth 42)
  all of them will be created automatically by data scripts if missing data files found
"""

import os
import pandas as pd
import numpy as np


# -------------------------------------------------
# ROOT FINDER
# -------------------------------------------------
def find_project_root(project_name="FixedToothDecayProject"):
    current_path = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.basename(current_path) == project_name:
            return current_path
        parent = os.path.dirname(current_path)
        if parent == current_path:
            raise FileNotFoundError(f"Project root '{project_name}' not found")
        current_path = parent


# -------------------------------------------------
# TOOTH-42 AXIS REMAPPING
# -------------------------------------------------
def build_axis_dict(data_dir):
    """
    Reads detected_grad_42.csv and returns a dict mapping
    (original_axis_1, original_axis_0) -> ((new_axis_0, new_axis_1), grad_value).
    Returns None if the file is missing.
    """
    csv_path = os.path.join(data_dir, "detected_grad_42.csv")
    if not os.path.exists(csv_path):
        print(f"WARNING: {csv_path} not found – tooth-42 axis remapping will be skipped.")
        return None

    extra = pd.read_csv(csv_path)
    # columns expected: [0]=something, [1]=something, [2]=axis_1, [3]=axis_0, [4]=grad
    axis_list = extra.apply(lambda x: ((x[2], x[3]), ((x[0], x[1]), x[4])), axis=1)
    return {key: value for key, value in axis_list}


def change_axis_and_label_for_42(df, axis_dict):
    """
    Remaps Axis_0 / Axis_1 for tooth 42 in-place and sets its Typ_zeba to 'Zdrowe'.
    Does nothing if axis_dict is None.
    """
    if axis_dict is None:
        return df

    mask = df["ID_zeba"] == 42
    if not mask.any():
        return df

    axis_0_new, axis_1_new, typ_list = [], [], []
    for _, row in df.loc[mask].iterrows():
        key = (row["Axis_1"], row["Axis_0"])
        if key not in axis_dict:
            axis_0_new.append(row["Axis_0"])
            axis_1_new.append(row["Axis_1"])
        else:
            ans_tuple, _ = axis_dict[key]
            axis_0_new.append(ans_tuple[1])
            axis_1_new.append(ans_tuple[0])
        typ_list.append("Zdrowe")

    df.loc[mask, "Axis_0"] = axis_0_new
    df.loc[mask, "Axis_1"] = axis_1_new
    df.loc[mask, "Typ_zeba"] = typ_list
    return df


# -------------------------------------------------
# MAIN LOADER
# -------------------------------------------------
def load_all_data(apply_axis_remap=True):
    """
    Returns a dict with all dataframes needed by the model scripts:

    Cleaned (intensity_at_XXX columns — for MiniRocket & Peak detection):
        'clean_aug'      – augmented, all polarisations
        'clean_nonaug'   – non-augmented, all polarisations

    Raw (Wavenumbers / Intensities columns — for DeepSets):
        'raw_aug'        – augmented, all polarisations
        'raw_nonaug'     – non-augmented, all polarisations

    Also returns:
        'data_dir'       – path to Data/ folder
        'tooth_image'    – path to tooth image (zab_og.png or image.png), or None
    """
    root = find_project_root()
    data_dir = os.path.join(root, "Data")

    axis_dict = build_axis_dict(data_dir) if apply_axis_remap else None

    def _load(fname, cast_id=True):
        path = os.path.join(data_dir, fname)
        df = pd.read_parquet(path, engine="pyarrow")
        if cast_id and "ID_zeba" in df.columns:
            df["ID_zeba"] = df["ID_zeba"].astype("int32")
        if apply_axis_remap:
            change_axis_and_label_for_42(df, axis_dict)
        return df

    print("Loading cleaned augmented data...")
    clean_aug = _load("scans_clean_augmented.parquet")

    print("Loading cleaned non-augmented data...")
    clean_nonaug = _load("scans_clean_nonaugmented.parquet")

    print("Loading raw augmented data...")
    raw_aug = _load("scans_augmented.parquet")

    print("Loading raw non-augmented data...")
    raw_nonaug = _load("scans_nonaugmented.parquet")

    # Locate tooth image (supports both naming conventions)
    tooth_image = None
    for candidate in ["zab_og.png", "image.png"]:
        p = os.path.join(root, candidate)
        if os.path.exists(p):
            tooth_image = p
            break
        p = os.path.join(data_dir, candidate)
        if os.path.exists(p):
            tooth_image = p
            break

    return {
        "clean_aug": clean_aug,
        "clean_nonaug": clean_nonaug,
        "raw_aug": raw_aug,
        "raw_nonaug": raw_nonaug,
        "data_dir": data_dir,
        "root": root,
        "tooth_image": tooth_image,
    }


# -------------------------------------------------
# SHARED METRIC
# -------------------------------------------------
def evaluate_model(y_test, y_pred, y_proba):
    """Weighted OvO ROC-AUC."""
    from sklearn.metrics import roc_auc_score
    n_classes = y_proba.shape[1]
    y_bin = np.stack([(y_test == i).astype(np.int32) for i in range(n_classes)], axis=1)
    if len(np.unique(y_bin)) <= 1:
        return "-----"
    return roc_auc_score(y_bin, y_proba, average="weighted", multi_class="ovo")