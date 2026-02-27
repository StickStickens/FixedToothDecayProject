import os
import numpy as np
import pandas as pd
from tqdm import tqdm
# old data standarisation file logic

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
# EXACT NOTEBOOK CLEANING LOGIC
# -------------------------------------------------
def clean_df_range(df, minv, maxv):

    waves = np.array(df["Wavenumbers"], dtype=object)

    df["minimum"] = [np.min(arr) for arr in waves]
    df["maximum"] = [np.max(arr) for arr in waves]

    # Keep only spectra covering full range
    df = df[(df["minimum"] < minv + 1) & (df["maximum"] > maxv - 1)].copy()

    lengths = np.array(df["Wavenumbers"], dtype=object)
    intensities = np.array(df["Intensities"], dtype=object)

    lengths = np.array([np.round(x) for x in lengths], dtype=object)

    # Mask range
    mask = [((x >= minv) & (x <= maxv)) for x in lengths]
    lengths = [x[m] for x, m in zip(lengths, mask)]
    intensities = [y[m] for y, m in zip(intensities, mask)]

    # Merge duplicates
    new_lengths, new_intensities = [], []

    for L, I in tqdm(zip(lengths, intensities), total=len(lengths)):
        idx = np.argsort(L)[::-1]
        L_sorted, I_sorted = L[idx], I[idx]

        merged_L, merged_I = [], []
        i = 0
        while i < len(L_sorted):
            j = i + 1
            while j < len(L_sorted) and L_sorted[j] == L_sorted[i]:
                j += 1
            merged_L.append(L_sorted[i])
            merged_I.append(np.mean(I_sorted[i:j]))
            i = j

        new_lengths.append(np.array(merged_L))
        new_intensities.append(np.array(merged_I))

    # Fill gaps
    new_lengths_filled, new_intensities_filled = [], []

    for L, I in tqdm(zip(new_lengths, new_intensities), total=len(new_lengths)):
        full_L = np.arange(minv, maxv + 1)
        full_I = np.zeros_like(full_L, dtype=float)

        L_sorted_idx = np.argsort(L)
        L_sorted, I_sorted = L[L_sorted_idx], I[L_sorted_idx]

        l_idx = 0
        for i, val in enumerate(full_L):
            if l_idx < len(L_sorted) and val == L_sorted[l_idx]:
                full_I[i] = I_sorted[l_idx]
                l_idx += 1
            else:
                prev_val = full_I[i - 1] if i > 0 else I_sorted[0]
                next_val = I_sorted[l_idx] if l_idx < len(I_sorted) else I_sorted[-1]
                full_I[i] = (prev_val + next_val) / 2

        new_lengths_filled.append(full_L)
        new_intensities_filled.append(full_I)

    intensities_2d = np.stack(new_intensities_filled)
    lengths_2d = np.stack(new_lengths_filled)

    # Expand to columns
    for i, arr in enumerate(intensities_2d.T):
        df[f"intensity_at_{lengths_2d[0][i]}"] = arr

    df = df.drop(
        columns=["Wavenumbers", "Intensities", "minimum", "maximum"],
        errors="ignore",
    )

    return df


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    print("starting cleaning data")
    root = find_project_root()
    data_dir = os.path.join(root, "Data")

    minv, maxv = 900, 990

    aug_path = os.path.join(data_dir, "scans_augmented.parquet")
    nonaug_path = os.path.join(data_dir, "scans_nonaugmented.parquet")

    df_aug = pd.read_parquet(aug_path, engine="pyarrow")
    df_nonaug = pd.read_parquet(nonaug_path, engine="pyarrow")

    df_clean_aug = clean_df_range(df_aug, minv, maxv)
    df_clean_nonaug = clean_df_range(df_nonaug, minv, maxv)

    print("cleaning augmented data")
    df_clean_aug.to_parquet(
        os.path.join(data_dir, "scans_clean_augmented.parquet"),
        engine="pyarrow",
        index=False,
    )

    print("cleaning nonaugmented data")
    df_clean_nonaug.to_parquet(
        os.path.join(data_dir, "scans_clean_nonaugmented.parquet"),
        engine="pyarrow",
        index=False,
    )

    print("Cleaning complete.")