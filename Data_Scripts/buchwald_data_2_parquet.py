import os
import pandas as pd
from collections import defaultdict
# old file readinng file

# -------------------------------------------------
# ROOT DIRECTORY FINDER
# -------------------------------------------------
def find_project_root(project_name="FixedToothDecayProject"):
    """
    Walks up from this file location until it finds the project root folder.
    """
    current_path = os.path.abspath(os.path.dirname(__file__))

    while True:
        if os.path.basename(current_path) == project_name:
            return current_path

        parent = os.path.dirname(current_path)
        if parent == current_path:
            raise FileNotFoundError(f"Could not find project root '{project_name}'")

        current_path = parent


# -------------------------------------------------
# FILE READING
# -------------------------------------------------
def read_file(file_path, is_single_place):
    """
    Reads a given file and returns data in format:
    [
        [pos_x_0, pos_y_0, [wavenumbers], [intensities]],
        ...
    ]
    """
    results = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = [line.strip().split() for line in lines if line.strip()]

    if is_single_place:
        wavenumbers = [float(d[0]) for d in data]
        intensities = [float(d[1]) for d in data]
        results.append([-1.0, -1.0, wavenumbers, intensities])
    else:
        grouped = defaultdict(lambda: {"w": [], "i": []})

        for d in data:
            x, y, w, i = map(float, d)
            grouped[(x, y)]["w"].append(w)
            grouped[(x, y)]["i"].append(i)

        for (x, y), vals in grouped.items():
            results.append([x, y, vals["w"], vals["i"]])

    return results


# -------------------------------------------------
# READ ALL SCANS
# -------------------------------------------------
def read_all_txt_scans(data_dir):
    folder_names = [
        "Chore_początkowo",
        "Chore_zaawansowanie",
        "Zdrowe",
        "Chore_sztucznie",
    ]

    columns = [
        "Typ_zeba",
        "ID_zeba",
        "Polaryzacja",
        "ID_skanu",
        "Is_single_place",
        "Axis_0",
        "Axis_1",
        "Wavenumbers",
        "Intensities",
        "time",
    ]

    df = pd.DataFrame(columns=columns)

    teeth_ordered_folder = os.path.join(data_dir,  "teeth_ordered_data")

    for folder in os.listdir(teeth_ordered_folder):
        print(f"Processing folder: {folder}")
        folder_path = os.path.join(teeth_ordered_folder, folder)

        if not os.path.isdir(folder_path):
            continue
        if folder not in folder_names:
            continue

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(".txt"):
                continue

            file_path = os.path.join(folder_path, fname)
            name, _ = os.path.splitext(fname)
            fsplit = name.split("_")

            content = read_file(file_path, fsplit[4] == "jeden")

            for one_scan in content:
                row = {
                    "Typ_zeba": folder,
                    "ID_zeba": fsplit[1],
                    "Polaryzacja": fsplit[2],
                    "ID_skanu": fsplit[3],
                    "Is_single_place": fsplit[4] == "jeden",
                    "Axis_0": one_scan[0],
                    "Axis_1": one_scan[1],
                    "Wavenumbers": one_scan[2],
                    "Intensities": one_scan[3],
                    "time": "None",
                }

                df.loc[len(df)] = row

    return df


# -------------------------------------------------
# LABEL MODIFICATION
# -------------------------------------------------
def change_label(row):
    if row["Typ_zeba"] in ["Chore_początkowo", "Chore_zaawansowanie"]:
        row["Typ_zeba"] = "Chore"
        return row

    if row["Typ_zeba"] == "Chore_sztucznie":
        if 1200 < row["Axis_1"] < 1800: # usuwanie zębów na granicy zdrowe/chore_sztucznie
            row["Typ_zeba"] = "Do_usuniecia"
            return row
        if row["Axis_1"] <= 1200:
            row["Typ_zeba"] = "Zdrowe"

    return row


# -------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------
if __name__ == "__main__":
    print("Starting data processing: file_reading.py...")

    # Find project root
    project_root = find_project_root()

    data_dir = os.path.join(project_root, "Data")
    os.makedirs(data_dir, exist_ok=True)

    # Read raw data
    df_raw = read_all_txt_scans(data_dir)

    # # Save raw data
    # raw_path = os.path.join(data_dir, "scans_raw.parquet")
    # df_raw.to_parquet(raw_path, engine="pyarrow", index=False)

    # Modify labels
    df_modified = df_raw.apply(change_label, axis=1)
    df_modified = df_modified.loc[df_modified["Typ_zeba"] != "Do_usuniecia"]

    # Save modified data
    modified_path = os.path.join(data_dir, "scans_nonaugmented.parquet")
    df_modified.to_parquet(modified_path, engine="pyarrow", index=False)

    print("Processing complete.")
    # print(f"Raw data saved to: {raw_path}")
    print(f"Modified data saved to: {modified_path}")