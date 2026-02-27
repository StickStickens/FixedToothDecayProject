"""
run_evaluation.py
-----------------
Orchestrates all three models, produces AUC summary tables (CSV) and
comparison heatmap plots for tooth 42.

Usage:
    python ML_Scripts/run_evaluation.py              # standard run
    python ML_Scripts/run_evaluation.py --no-plots   # tables only (faster)
    python ML_Scripts/run_evaluation.py --epochs 50  # change DeepSets epochs

Output (written to Results/):
    Results/
    ├── tables/
    │   ├── Chore_Zdrowe_table.csv
    │   ├── Chore_sztucznie_Zdrowe_table.csv
    │   └── all_classes_table.csv
    └── plots/
        └── Chore_Zdrowe_comparison.png
        └── …
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_all_data, evaluate_model
from model_peak_detection import peak_classifier
from model_minirocket import predict_with_minirocket
from model_deepsets import predict_with_deepsets


# =============================================================
# HEATMAP PLOT
# =============================================================

def _make_heatmap_field(df, grid_res=400):
    x = df["Axis_1"].values
    y = df["Axis_0"].values
    z = df["predicted"].values.astype(float)
    z_min, z_max = z.min(), z.max()
    z = (z - z_min) / (z_max - z_min + 1e-12)

    XMIN, XMAX, YMIN, YMAX = 0, 3500, 0, 3000
    xi = np.linspace(XMIN, XMAX, grid_res)
    yi = np.linspace(YMIN, YMAX, grid_res)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method="cubic")
    zi = gaussian_filter(zi, sigma=5)
    return xi, yi, zi


def plot_comparison_2x2(df_minirocket, df_peak, df_deepsets,
                        tooth_image_path, title="Comparison of methods",
                        file_path=None, cmap="magma", show_contours=True):
    """
    2×2 panel: top-left = tooth photo, others = heatmaps for each model.
    Any of df_minirocket / df_peak / df_deepsets may be None (panel skipped).
    """
    XMIN, XMAX, YMIN, YMAX = 0, 3500, 0, 3000

    panels = [
        ("MiniRocket", df_minirocket),
        ("XGBoost peak", df_peak),
        ("Deep Sets", df_deepsets),
    ]

    fields = {}
    for name, df in panels:
        if df is not None and not df.empty:
            fields[name] = _make_heatmap_field(df)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.ravel()
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.06,
                        top=0.90, wspace=0.25, hspace=0.12)

    # Panel 0 – tooth image
    if tooth_image_path and os.path.exists(tooth_image_path):
        img = mpimg.imread(tooth_image_path)
        axs[0].imshow(img, extent=[XMIN, XMAX, YMIN, YMAX], origin="upper")
    else:
        axs[0].set_facecolor("#cccccc")
        axs[0].text(XMAX // 2, YMAX // 2, "No tooth image found",
                    ha="center", va="center", fontsize=10)
    axs[0].set_title("Tooth image")
    axs[0].set_xlabel("X/μm"); axs[0].set_ylabel("Y/μm")
    axs[0].set_xlim(XMIN, XMAX); axs[0].set_ylim(YMIN, YMAX)
    axs[0].set_aspect("equal", adjustable="box")

    hm_last = None
    for i, (name, _) in enumerate(panels, start=1):
        ax = axs[i]
        if name not in fields:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(name)
            continue
        xi, yi, zi = fields[name]
        hm_last = ax.imshow(
            zi, extent=[XMIN, XMAX, YMIN, YMAX], origin="lower",
            cmap=cmap, alpha=0.85, vmin=0.0, vmax=1.0,
        )
        if show_contours:
            zi_c = np.where(zi > 0.2, zi, 0)
            ax.contour(xi, yi, zi_c, levels=10, colors="black",
                       linewidths=0.6, alpha=0.6)
        ax.set_title(name); ax.set_xlabel("X/μm"); ax.set_ylabel("Y/μm")
        ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX)
        ax.set_aspect("equal", adjustable="box")

    if hm_last is not None:
        cbar = fig.colorbar(hm_last, ax=axs.tolist(), shrink=0.85)
        cbar.set_label("Predicted value (normalised)")

    fig.suptitle(title, fontsize=14)

    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved → {file_path}")
    else:
        plt.show()


# =============================================================
# TABLE CREATION
# =============================================================

def build_results_table(classes, data, deepsets_epochs=50, skip_deepsets=False):
    """
    Returns a DataFrame with rows = methods and columns = polarisation configs.

    Columns (multi-index):
        (not augmented | augmented) × (v | vh+vv | v+vh+vv)
    """
    clean_aug = data["clean_aug"]
    clean_nonaug = data["clean_nonaug"]
    raw_aug = data["raw_aug"]

    col_tuples = [
        ("not augmented", "v"),
        ("not augmented", "vh + vv"),
        ("not augmented", "v + vh + vv"),
        ("augmented", "v"),
        ("augmented", "vh + vv"),
        ("augmented", "v + vh + vv"),
    ]
    pol_options = [["v"], ["vh", "vv"], ["vh", "vv", "v"]] * 2
    aug_flags   = [False, False, False, True, True, True]
    df_peak_src = [clean_nonaug, clean_nonaug, clean_nonaug,
                   clean_aug,    clean_aug,    clean_aug]
    df_mini_src = df_peak_src

    results = {}

    # --- Peak detection ---
    print("\n  [Peak detection]")
    row = []
    for df_src, aug, pols in zip(df_peak_src, aug_flags, pol_options):
        y_test, y_pred, y_proba = peak_classifier(
            df_src, augmented=aug, polarizations=pols,
            classes=classes, to_plot_data_42=False,
        )
        row.append(evaluate_model(y_test, y_pred, y_proba) if y_test is not None else "-------")
    results["peak_classifier"] = row

    # --- MiniRocket ---
    print("\n  [MiniRocket]")
    row = []
    for df_src, aug, pols in zip(df_mini_src, aug_flags, pol_options):
        y_test, y_pred, y_proba = predict_with_minirocket(
            df_src, augmented=aug, polarizations=pols,
            classes=classes, to_plot_data_42=False,
        )
        row.append(evaluate_model(y_test, y_pred, y_proba) if y_test is not None else "-------")
    results["minirocket"] = row

    # --- Deep Sets ---
    if not skip_deepsets:
        print("\n  [Deep Sets]")
        ds_row = []
        for i, (aug, pols) in enumerate(zip(aug_flags, pol_options)):
            df_src = raw_aug if aug else data["raw_nonaug"]
            y_test, y_pred, y_proba = predict_with_deepsets(
                df_src, epochs=deepsets_epochs,
                classes=classes, polarizations=pols, to_plot_data_42=False,
            )
            ds_row.append(evaluate_model(y_test, y_pred, y_proba) if y_test is not None else "-------")
        results["deepsets"] = ds_row

    final_df = pd.DataFrame(
        results,
        index=pd.MultiIndex.from_tuples(col_tuples),
    ).T
    final_df.columns = pd.MultiIndex.from_tuples(col_tuples)
    return final_df



# =============================================================
# FINAL COMPARISON PLOTS
# =============================================================

# All polarisation configs — used for both tables and plots
POLARIZATION_OPTIONS = [
    ["v"],
    ["vh", "vv"],
    ["vh", "vv", "v"],
]


def build_comparison_plot(classes, polarizations, data, deepsets_epochs=50, output_dir=None):
    """
    Runs each model in tooth-42 prediction mode for a given polarisation
    config and generates a 2x2 heatmap (tooth photo + 3 model heatmaps).

    DeepSets only supports single-channel input ('v'), so for multi-pol
    configs the DeepSets panel is omitted (shown as "No data").

    Parameters
    ----------
    classes       : list[str]
    polarizations : list[str]  e.g. ['v'] or ['vh', 'vv'] or ['vh', 'vv', 'v']
    data          : dict from load_all_data()
    deepsets_epochs : int
    output_dir    : str | None
    """
    if len(classes) == 3:
        return
    class_tag  = "_".join(classes)
    pol_tag    = "_".join(polarizations)   # e.g. "v", "vh_vv", "vh_vv_v"
    output_dir = output_dir or os.path.join(data["root"], "Results", "plots")
    file_path  = os.path.join(output_dir, f"{class_tag}_{pol_tag}_comparison.png")

    print(f"\n  Heatmap — classes={classes}  pol={polarizations}")

    print("    Running MiniRocket for tooth 42 ...")
    df_mini = predict_with_minirocket(
        data["clean_aug"], augmented=True, polarizations=polarizations,
        classes=classes, to_plot_data_42=True,
    )

    print("    Running Peak classifier for tooth 42 ...")
    df_peak = peak_classifier(
        data["clean_aug"], augmented=True, polarizations=polarizations,
        classes=classes, to_plot_data_42=True,
    )

    # DeepSets works on raw single-channel spectra — skip for multi-pol configs
    print("    Running Deep Sets for tooth 42 ...")
    df_deep = predict_with_deepsets(
        data["raw_aug"], epochs=deepsets_epochs,
        classes=classes, polarizations=polarizations, to_plot_data_42=True,
    )

    pol_label = " + ".join(polarizations)
    plot_comparison_2x2(
        df_mini, df_peak, df_deep,
        tooth_image_path=data["tooth_image"],
        title=f"Method comparison - {class_tag}  [{pol_label}]",
        file_path=file_path,
    )


# =============================================================
# MAIN
# =============================================================

def main(args):
    print("Loading data ...")
    data = load_all_data()

    root        = data["root"]
    results_dir = os.path.join(root, "Results")
    tables_dir  = os.path.join(results_dir, "tables")
    plots_dir   = os.path.join(results_dir, "plots")
    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    class_configs = [
        (["Chore", "Zdrowe"],                   "Chore_Zdrowe_table.csv"),
        (["Chore_sztucznie", "Zdrowe"],          "Chore_sztucznie_Zdrowe_table.csv"),
        (["Chore", "Zdrowe", "Chore_sztucznie"], "all_classes_table.csv"),
    ]

    for classes, csv_name in class_configs:
        print(f"\n{'='*60}")
        print(f"Class set: {classes}")

        table = build_results_table(
            classes, data,
            deepsets_epochs=args.epochs,
            skip_deepsets=args.no_deepsets,
        )
        csv_path = os.path.join(tables_dir, csv_name)
        table.to_csv(csv_path)
        print(f"\n  Table saved -> {csv_path}")
        print(table.to_string())

        if not args.no_plots:
            for pols in POLARIZATION_OPTIONS:
                build_comparison_plot(
                    classes, pols, data,
                    deepsets_epochs=args.epochs,
                    output_dir=plots_dir,
                )

    print("\n Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full model evaluation.")
    parser.add_argument("--no-plots",    action="store_true",
                        help="Skip tooth-42 comparison plots (faster).")
    parser.add_argument("--no-deepsets", action="store_true",
                        help="Skip Deep Sets model.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs for Deep Sets training (default: 50).")
    args = parser.parse_args()
    main(args)

### runing scripts:
#     # Full evaluation (all 3 models, all class combos, plots)
# python ML_Scripts/run_evaluation.py

# # Tables only, skip Deep Sets training (much faster for quick checks)
# python ML_Scripts/run_evaluation.py --no-plots --no-deepsets

# # Control Deep Sets training length
# python ML_Scripts/run_evaluation.py --epochs 100