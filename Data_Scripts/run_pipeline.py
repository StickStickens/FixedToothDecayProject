import os
import sys
import subprocess
import argparse


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
# RUN SCRIPT HELPER
# -------------------------------------------------
def run_script(script_path):
    print(f"\n▶ Running: {os.path.basename(script_path)}")
    subprocess.run([sys.executable, script_path], check=True)


# -------------------------------------------------
# MAIN PIPELINE LOGIC
# -------------------------------------------------
def main(force_all=False):

    root = find_project_root()
    data_dir = os.path.join(root, "Data")
    scripts_dir = os.path.join(root, "Data_Scripts")

    teeth_dir = os.path.join(data_dir, "teeth_ordered_data")

    scans_nonaug = os.path.join(data_dir, "scans_nonaugmented.parquet")
    scans_aug = os.path.join(data_dir, "scans_augmented.parquet")
    scans_clean_aug = os.path.join(data_dir, "scans_clean_augmented.parquet")
    scans_clean_nonaug = os.path.join(data_dir, "scans_clean_nonaugmented.parquet")

    # Required scripts
    script_buchwald = os.path.join(scripts_dir, "buchwald_data_2_parquet.py")
    script_aug = os.path.join(scripts_dir, "data_augmentation.py")
    script_clean = os.path.join(scripts_dir, "data_cleaning.py")

    sth_missing = 0 # if any step is missing, this will be set to 1 and all following steps will be run to ensure consistency. If all files are present, this remains 0 and all steps are skipped.
    # -------------------------------------------------
    # Check raw data folder
    # -------------------------------------------------
    if not os.path.exists(teeth_dir) or sth_missing == 1:
        print("No teeth_ordered_data found. Nothing can be done.")
        sth_missing = 1
        return

    # -------------------------------------------------
    # Step 1 — Generate nonaugmented parquet
    # -------------------------------------------------
    if force_all or not os.path.exists(scans_nonaug) or sth_missing == 1:
        run_script(script_buchwald)
        sth_missing = 1

    else:
        print("✔ scans_nonaugmented.parquet exists — skipping.")

    # -------------------------------------------------
    # Step 2 — Augmentation
    # -------------------------------------------------
    if force_all or not os.path.exists(scans_aug) or sth_missing == 1:
        run_script(script_aug)
        sth_missing = 1
    else:
        print("✔ scans_augmented.parquet exists — skipping.")

    # -------------------------------------------------
    # Step 3 — Cleaning
    # -------------------------------------------------
    clean_missing = (
        not os.path.exists(scans_clean_aug)
        or not os.path.exists(scans_clean_nonaug)
    )

    if force_all or clean_missing or sth_missing == 1:
        run_script(script_clean)
        sth_missing = 1
    else:
        print("✔ Clean parquet files exist — skipping.")

    print("\nPipeline finished.")


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data processing pipeline.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Force run all steps (ignore existing files)",
    )

    args = parser.parse_args()

    main(force_all=args.all)


# to run all use:
#  python Data_Scripts/run_pipeline.py --all