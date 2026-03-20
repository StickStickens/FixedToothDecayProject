import os
import pandas as pd
from collections import defaultdict
from typing import Any
from ..Utils.utils import find_project_root
# old file readinng file




def read_file(file_path: str, is_single_place: bool) -> list[list[Any]]:
    """Read a single scan text file and parse spectral data.

    Parameters
    ----------
    file_path : str
        Path to the scan `.txt` file.
    is_single_place : bool
        Whether the file contains one scan position (`jeden`) or many
        positions (`wiele`).

    Returns
    -------
    list[list[Any]]
        Parsed scans where each element has form:
        `[axis_0, axis_1, wavenumbers, intensities]`.
    """
    results: list[list[Any]] = []

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


def read_all_txt_scans(data_dir: str) -> pd.DataFrame:
    """Read all scans from `Data/teeth_ordered_data` into one DataFrame.

    Expected directory layout
    -------------------------
    ::

        Data/
            teeth_ordered_data/
                Chore_poczatkowo/
                    scan_001_jeden_v_1.txt
                    ...
                Chore_zaawansowanie/
                    scan_003_jeden_vv_1.txt
                    ...
                Zdrowe/
                    scan_004_jeden_v_1.txt
                    ...
                Chore_sztucznie/
                    scan_005_wiele_vh_1.txt
                    ...

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the tooth scan folders.

    Returns
    -------
    pandas.DataFrame
        Table with columns: `Typ_zeba`, `ID_zeba`, `Polaryzacja`,
        `ID_skanu`, `Is_single_place`, `Axis_0`, `Axis_1`, `Wavenumbers`,
        `Intensities`, and `time`.
    """

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



def change_label(row: pd.Series) -> pd.Series:
    
    """Remap `Typ_zeba` label for one row using Buchwald rules.

    Parameters
    ----------
    row : pandas.Series
        DataFrame row with `Typ_zeba` and `Axis_1` values.

    Returns
    -------
    pandas.Series
        Row with possibly updated `Typ_zeba`.

    Note
    ----
    Buchwald rules:
    - If "Typ_zeba" is "Chore_początkowo" or "Chore_zaawansowanie", change it to "Chore".
        - If "Typ_zeba" is "Chore_sztucznie":
            - If "Axis_1" is between 1200 and 1800, change "Typ_zeba" to "Do_usuniecia".
            - If "Axis_1" is less than or equal to 1200, change "Typ_zeba" to "Zdrowe".
        It  is based on the information provided by dr. Buchwald, based on the photograph of a tooth with a scan axis.
    """
    if row["Typ_zeba"] in ["Chore_początkowo", "Chore_zaawansowanie"]:
        row["Typ_zeba"] = "Chore"
        return row

    if row["Typ_zeba"] == "Chore_sztucznie":
        if 1200 < row["Axis_1"] < 1800: # TO ensure proper labels, we will remove scans from the area between sane and artificially decayed part
            row["Typ_zeba"] = "Do_usuniecia"
            return row
        if row["Axis_1"] <= 1200:
            row["Typ_zeba"] = "Zdrowe"

    return row



# Execution of the program starts here
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