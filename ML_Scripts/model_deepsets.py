"""
model_deepsets.py
-----------------
Deep Sets neural network for Raman spectra classification.

Works on the RAW parquet files (Wavenumbers / Intensities columns).
Detects significant peaks per spectrum, encodes each peak as a feature
vector {wavenumber, height, prominence, width, snr, pol_id}, and passes
the variable-length set through a permutation-invariant Deep Sets network.

Multi-polarisation support: when multiple polarisations are requested,
peaks from each polarisation are detected independently and concatenated
into one set. Each peak carries a `pol_id` integer (0, 1, 2, ...) so the
network can distinguish which polarisation a peak came from. The set size
cap (max_peaks) scales with the number of polarisations automatically.

Matching the original notebook: the original only ever used 'v' — this
version is fully backwards-compatible with that behaviour.

Can be run standalone:
    python ML_Scripts/model_deepsets.py

Or imported and called programmatically (used by run_evaluation.py).
"""

import warnings
warnings.filterwarnings("ignore")

import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.signal import savgol_filter, find_peaks
from tqdm import tqdm

from data_loader import load_all_data, evaluate_model


# =============================================================
# REPRODUCIBILITY
# =============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

FEATURE_KEYS_BASE = ["wavenumber", "height", "prominence", "width", "snr"]
MAX_PEAKS_PER_POL = 5   # cap per polarisation channel; total = MAX_PEAKS_PER_POL x n_pols

# Peak detection hyper-parameters
WINDOW_LENGTH = 11
POLYORDER = 3
MIN_PEAK_DISTANCE_CM = 100
REL_PROMINENCE = 0.05


# =============================================================
# PEAK DETECTION
# =============================================================

def find_significant_peaks(row):
    """
    Detect significant peaks in a raw spectrum row.
    Returns (list of peak-feature dicts, smoothed intensity array).

    Accepts rows with either:
      - 'Wavenumbers' + 'Intensities' columns  (raw parquet)
      - 'intensity_at_XXX' columns             (cleaned parquet)
    """
    if "Wavenumbers" in row.index and "Intensities" in row.index:
        w = np.array(row["Wavenumbers"])
        I = np.array(row["Intensities"])
    else:
        intensity_cols = [c for c in row.index if str(c).startswith("intensity_at_")]
        if not intensity_cols:
            return [], np.array([])
        w = np.array([int(c.split("_")[-1]) for c in intensity_cols])
        I = np.array(row[intensity_cols], dtype=float)

    I_smooth = savgol_filter(I, window_length=WINDOW_LENGTH, polyorder=POLYORDER)
    prom_value = I_smooth.max() * REL_PROMINENCE
    peaks, properties = find_peaks(I_smooth, prominence=prom_value, width=1)

    sorted_peaks = peaks[np.argsort(I_smooth[peaks])[::-1]]
    final = []
    for p in sorted_peaks:
        if all(abs(w[p] - w[fp]) >= MIN_PEAK_DISTANCE_CM for fp in final):
            final.append(p)
    final = np.array(sorted(final, key=lambda idx: w[idx]))

    peak_list = []
    for p in final:
        idx        = np.where(peaks == p)[0][0]
        height     = float(I_smooth[p])
        prominence = float(properties["prominences"][idx])
        width      = float(properties["widths"][idx])
        bg_mask    = abs(w - w[p]) > 30
        background = float(np.mean(I_smooth[bg_mask])) if bg_mask.any() else 0.0
        snr        = height / (background + 1e-9)
        peak_list.append({
            "wavenumber": float(w[p]),
            "height":     height,
            "prominence": prominence,
            "width":      width,
            "snr":        snr,
        })

    return peak_list, I_smooth


# =============================================================
# MULTI-POLARISATION FEATURE EXTRACTION
# =============================================================

def _extract_features_multipol(df, polarizations):
    """
    For each unique scan position (ID_zeba, ID_skanu, Axis_0, Axis_1),
    collect peaks from every requested polarisation, tag each peak with a
    numeric pol_id, and concatenate into one set.

    When only one polarisation is requested, pol_id is NOT added (matches
    original notebook behaviour, 5 features per peak instead of 6).

    Returns
    -------
    X            : list[list[dict]]  -- one entry per scan position
    y            : list[str]         -- label
    meta         : list[dict]        -- ID_zeba, Axis_0, Axis_1 for plotting
    feature_keys : list[str]         -- ordered feature names used
    """
    use_pol_id   = len(polarizations) > 1
    feature_keys = FEATURE_KEYS_BASE + (["pol_id"] if use_pol_id else [])

    # Build lookup: position_key -> {pol: row}
    pol_dicts = {}
    for pol in polarizations:
        sub = df[df["Polaryzacja"] == pol]
        for _, row in sub.iterrows():
            key = (row["ID_zeba"], row["ID_skanu"], row["Axis_0"], row["Axis_1"])
            pol_dicts.setdefault(key, {})[pol] = row

    X, y, meta = [], [], []
    for key in sorted(pol_dicts.keys()):
        pol_rows       = pol_dicts[key]
        combined_peaks = []

        for pol_idx, pol in enumerate(polarizations):
            if pol not in pol_rows:
                continue
            peaks, _ = find_significant_peaks(pol_rows[pol])
            for peak in peaks:
                entry = {k: peak[k] for k in FEATURE_KEYS_BASE}
                if use_pol_id:
                    entry["pol_id"] = float(pol_idx)
                combined_peaks.append(entry)

        ref_row = next(iter(pol_rows.values()))
        X.append(combined_peaks)
        y.append(ref_row["Typ_zeba"])
        meta.append({
            "ID_zeba": ref_row["ID_zeba"],
            "Axis_0":  ref_row["Axis_0"],
            "Axis_1":  ref_row["Axis_1"],
        })

    return X, y, meta, feature_keys


# =============================================================
# PYTORCH DATASET
# =============================================================

class PeakDataset(Dataset):
    def __init__(self, X_list, y_array, feature_keys, max_peaks):
        self.max_peaks    = max_peaks
        self.num_features = len(feature_keys)
        self.feature_keys = feature_keys
        self.X = []
        self.y = torch.tensor(y_array, dtype=torch.long)

        for peaks in X_list:
            if peaks:
                arr = np.array(
                    [[p[k] for k in feature_keys] for p in peaks],
                    dtype=np.float32,
                )
            else:
                arr = np.zeros((0, self.num_features), dtype=np.float32)
            self.X.append(arr)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        peaks = self.X[idx]
        n     = peaks.shape[0]
        if n < self.max_peaks:
            pad   = np.zeros((self.max_peaks - n, self.num_features), dtype=np.float32)
            peaks = np.vstack([peaks, pad]) if n > 0 else pad
        else:
            peaks = peaks[: self.max_peaks]
        return torch.tensor(peaks, dtype=torch.float32), self.y[idx]


# =============================================================
# BALANCED BATCH SAMPLER
# =============================================================

class BalancedBatchSampler:
    def __init__(self, idx_classes, batch_size, num_batches):
        self.idx_classes = idx_classes
        self.batch_size  = batch_size
        self.per_class   = batch_size // len(idx_classes)
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for idx_class in self.idx_classes:
                batch.extend(
                    np.random.choice(
                        idx_class, self.per_class,
                        replace=len(idx_class) < self.per_class,
                    )
                )
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# =============================================================
# DEEP SETS MODEL
# =============================================================

class DeepSetsImproved(nn.Module):
    def __init__(self, num_features=5, phi_dim=32, rho_dim=32,
                 num_classes=3, emb_dim=4, dropout=0.45):
        super().__init__()
        self.num_features = num_features
        self.no_peak_emb  = nn.Parameter(torch.randn(1, 1, num_features) * 0.01)
        self.feature_embs = nn.ModuleList(
            [nn.Linear(1, emb_dim) for _ in range(num_features)]
        )
        self.phi = nn.Sequential(
            nn.LayerNorm(num_features * emb_dim),
            nn.Linear(num_features * emb_dim, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, phi_dim),
        )
        self.att_gate = nn.Sequential(
            nn.Linear(phi_dim, phi_dim // 4),
            nn.SiLU(),
            nn.Linear(phi_dim // 4, 1),
            nn.Sigmoid(),
        )
        self.rho = nn.Sequential(
            nn.LayerNorm(phi_dim),
            nn.Linear(phi_dim, rho_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(rho_dim, num_classes),
        )

    def forward(self, x):
        mask    = (x.abs().sum(dim=2, keepdim=True) != 0).float()
        x       = x * mask + self.no_peak_emb * (1 - mask)
        emb     = torch.cat(
            [self.feature_embs[i](x[:, :, i: i + 1]) for i in range(self.num_features)],
            dim=2,
        )
        phi_out = self.phi(emb)
        att     = self.att_gate(phi_out) * mask
        phi_out = phi_out * att
        pooled  = phi_out.sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        return self.rho(pooled), phi_out


# =============================================================
# TRAIN / PREDICT
# =============================================================

def predict_with_deepsets(
    df,
    epochs=100,
    classes=None,
    polarizations=None,
    to_plot_data_42=False,
):
    """
    Train a Deep Sets model on df, evaluate, and optionally return
    tooth-42 predictions for heatmap plotting.

    Parameters
    ----------
    df            : DataFrame -- raw parquet (Wavenumbers / Intensities columns)
    epochs        : int
    classes       : list[str] | None
    polarizations : list[str] | None
        Defaults to ['v'] to match the original notebook.
        For multi-pol, peaks from each channel are merged into one set
        and each peak receives an extra pol_id feature.
    to_plot_data_42 : bool
        If True  -> returns df_42 with 'predicted' column (Axis_0, Axis_1).
        If False -> returns (y_test, y_pred, y_proba).
    """
    polarizations = polarizations or ["v"]
    n_pols        = len(polarizations)
    max_peaks     = MAX_PEAKS_PER_POL * n_pols   # 5, 10, or 15

    # Separate tooth-42 before any class filtering
    df_42_full = df[df["ID_zeba"] == 42].reset_index(drop=True)

    if classes:
        df = df[df["Typ_zeba"].isin(classes)].reset_index(drop=True)

    df_no42 = df[df["ID_zeba"] != 42].reset_index(drop=True)

    print("classes", classes)
    print("comparison", df_no42["Typ_zeba"].nunique(), len(classes))
    if df_no42["Typ_zeba"].nunique() < len(classes) if classes else 2:
        return (None, None, None) if not to_plot_data_42 else None

    # --- Feature extraction ---
    print(f"  DeepSets: extracting peaks from {polarizations} ...")
    X_all, y_all, _,       feature_keys = _extract_features_multipol(df_no42,    polarizations)
    X_42,  y_42,  meta_42, _            = _extract_features_multipol(df_42_full, polarizations)

    num_features = len(feature_keys)

    # --- Labels ---
    le       = LabelEncoder()
    y_enc    = le.fit_transform(y_all)
    le_42    = LabelEncoder().fit(y_42)
    y_42_enc = le_42.transform(y_42)

    # --- Train / test split ---
    indices        = np.arange(len(X_all))
    tr_idx, te_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_train     = [X_all[i] for i in tr_idx];  y_train_enc = y_enc[tr_idx]
    X_test      = [X_all[i] for i in te_idx];  y_test_enc  = y_enc[te_idx]

    train_ds   = PeakDataset(X_train, y_train_enc, feature_keys, max_peaks)
    test_ds    = PeakDataset(X_test,  y_test_enc,  feature_keys, max_peaks)
    test_42_ds = PeakDataset(X_42,    y_42_enc,    feature_keys, max_peaks)

    # --- Balanced sampler ---
    classes_codes = np.unique(y_train_enc)
    idx_classes   = [np.where(y_train_enc == c)[0] for c in classes_codes]
    idx_classes   = [idx for idx in idx_classes if len(idx) > 0]
    num_batches   = max(1, len(train_ds) // BATCH_SIZE)

    if len(idx_classes) >= 2:
        sampler      = BalancedBatchSampler(idx_classes, BATCH_SIZE, num_batches)
        train_loader = DataLoader(train_ds, batch_sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    test_loader    = DataLoader(test_ds,    batch_size=BATCH_SIZE,      shuffle=False)
    test_42_loader = DataLoader(test_42_ds, batch_size=len(test_42_ds), shuffle=False)

    # --- Model (num_features adapts to pol count) ---
    model     = DeepSetsImproved(
                    num_features=num_features,
                    num_classes=len(le.classes_),
                ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    print(f"  Training Deep Sets for {epochs} epochs on {DEVICE}  "
          f"(features/peak={num_features}, max_peaks={max_peaks}) ...")
    for epoch in range(epochs):
        model.train()
        total_loss, all_preds, all_labels = 0.0, [], []
        for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(Xb)
            loss      = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
            all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
            all_labels.extend(yb.detach().cpu().tolist())

        if epoch == epochs - 1:
            avg_loss = total_loss / len(train_ds)
            acc      = accuracy_score(all_labels, all_preds)
            f1       = f1_score(all_labels, all_preds, average="macro")
            print(f"  Epoch {epoch+1} -- Loss:{avg_loss:.4f} | Acc:{acc:.4f} | F1_macro:{f1:.4f}")

    # --- Evaluation ---
    model.eval()
    all_preds, all_labels, all_probas = [], [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb    = Xb.to(DEVICE)
            logits, _ = model(Xb)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            all_labels.extend(yb.numpy())
            all_probas.extend(probs.cpu().numpy())

    if not to_plot_data_42:
        return (
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probas),
        )

    # --- Tooth-42 predictions ---
    disease_class        = 1 if le_42.classes_[0] == "Zdrowe" else 0
    test_probs, rows_keep = [], []

    with torch.no_grad():
        for Xb, _ in test_42_loader:
            Xb           = Xb.to(DEVICE)
            nonzero_mask = (Xb.abs().sum(dim=2) != 0).any(dim=1)
            if nonzero_mask.sum() == 0:
                continue
            logits, _ = model(Xb)
            probs      = torch.softmax(logits, dim=1)
            preds      = torch.argmax(probs, dim=1)
            if len(le.classes_) == 3:
                test_probs.extend(probs[nonzero_mask].cpu().numpy())
            else:
                test_probs.extend(preds[nonzero_mask].cpu().numpy())
            rows_keep.extend(torch.where(nonzero_mask)[0].tolist())

    df_42_out = pd.DataFrame([meta_42[i] for i in rows_keep])

    preds_col = []
    for prob in test_probs:
        p = np.atleast_1d(prob)
        preds_col.append(p[disease_class] if len(p) > disease_class else float(prob))

    df_42_out["predicted"] = 1 - np.array(preds_col)
    return df_42_out


# =============================================================
# STANDALONE
# =============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    data    = load_all_data()
    raw_aug = data["raw_aug"]

    class_sets = [
        ["Chore", "Zdrowe"],
        ["Chore_sztucznie", "Zdrowe"],
    ]
    polarization_options = [["v"], ["vh", "vv"], ["vh", "vv", "v"]]

    for classes in class_sets:
        print(f"\n{'='*60}")
        print(f"Classes: {classes}")
        for pols in polarization_options:
            print(f"\n  Polarisations: {pols}")
            y_test, y_pred, y_proba = predict_with_deepsets(
                raw_aug, epochs=50, classes=classes,
                polarizations=pols, to_plot_data_42=False,
            )
            if y_test is None:
                print("  -------")
            else:
                auc = evaluate_model(y_test, y_pred, y_proba)
                print(f"  AUC={auc:.4f}" if isinstance(auc, float) else f"  {auc}")