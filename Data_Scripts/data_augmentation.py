import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


GLOBAL_SEED = 42


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
# AUGMENTATION FUNCTIONS
# -------------------------------------------------
def add_noise(intensity, rng, noise_std=0.01):
    return intensity + rng.normal(0, noise_std, size=intensity.shape)


def baseline_drift(wavenumbers, intensity, rng, drift_scale=0.02):
    freq = rng.uniform(0.5, 2)
    drift = drift_scale * np.sin(
        wavenumbers / wavenumbers.max() * 2 * np.pi * freq
    )
    return intensity + drift


def scale_intensity(intensity, rng, scale_range=(0.9, 1.1)):
    scale = rng.uniform(*scale_range)
    return intensity * scale


def shift_spectrum(wavenumbers, intensity, rng, shift_range=(-3, 3)):
    shift = rng.uniform(*shift_range)
    f = interp1d(
        wavenumbers + shift,
        intensity,
        bounds_error=False,
        fill_value="extrapolate",
    )
    return f(wavenumbers)


def broaden_peaks(intensity, rng, sigma_range=(0.5, 2.0)):
    sigma = rng.uniform(*sigma_range)
    return gaussian_filter1d(intensity, sigma)


def kumaraswamy_warp(wavenumbers, intensity, rng, a_range=(0.8, 2.0), b_range=(0.8, 2.0)):
    a = rng.uniform(*a_range)
    b = rng.uniform(*b_range)

    wn_norm = (wavenumbers - wavenumbers.min()) / (
        wavenumbers.max() - wavenumbers.min()
    )

    warped_norm = 1 - (1 - wn_norm ** a) ** b
    warped_wn = wavenumbers.min() + warped_norm * (
        wavenumbers.max() - wavenumbers.min()
    )

    f = interp1d(
        wavenumbers,
        intensity,
        bounds_error=False,
        fill_value="extrapolate",
    )

    return f(warped_wn)


def augment_spectrum(wavenumbers, intensity, base_seed):
    rng = np.random.default_rng(base_seed)

    return {
        "noise": add_noise(intensity, rng),
        "baseline": baseline_drift(wavenumbers, intensity, rng),
        "scaled": scale_intensity(intensity, rng),
        "shifted": shift_spectrum(wavenumbers, intensity, rng),
        "broadened": broaden_peaks(intensity, rng),
        "kumaraswamy": kumaraswamy_warp(wavenumbers, intensity, rng),
    }


# -------------------------------------------------
# DATAFRAME AUGMENTATION
# -------------------------------------------------
def augment_raman_dataframe(df):

    augmented_rows = []

    mask = (
        (df["Typ_zeba"].str.lower() != "zdrowe")
        & (df["ID_zeba"] != "042")
        & (df["Typ_zeba"] != "Chore_sztucznie")
    )

    df_original = df.copy()
    df_original["augmentation_type"] = "original"

    for idx, row in df[mask].iterrows():
        wn = np.array(row["Wavenumbers"])
        intensity = np.array(row["Intensities"])

        # Deterministic seed per row
        row_seed = GLOBAL_SEED + hash((row["ID_zeba"], row["ID_skanu"])) % 10_000

        augmented = augment_spectrum(wn, intensity, row_seed)

        for aug_type, aug_intensity in augmented.items():
            new_row = row.copy()
            new_row["Intensities"] = aug_intensity.tolist()
            new_row["augmentation_type"] = aug_type
            augmented_rows.append(new_row)

    df_augmented = pd.DataFrame(augmented_rows)

    df_final = pd.concat([df_original, df_augmented], ignore_index=True)
    df_final = df_final.reset_index(drop=True)

    return df_final


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    print("Augmentating data...")
    root = find_project_root()
    data_dir = os.path.join(root, "Data")

    input_path = os.path.join(data_dir, "scans_nonaugmented.parquet")
    output_path = os.path.join(data_dir, "scans_augmented.parquet")

    df = pd.read_parquet(input_path, engine="pyarrow")

    df_augmented = augment_raman_dataframe(df)

    df_augmented.to_parquet(output_path, engine="pyarrow", index=False)

    print("Augmentation complete.")