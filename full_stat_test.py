#!/usr/bin/env python3
import os
import sys
import pandas as pd
import subprocess
from pathlib import Path

def process_csvs(folder):
    folder = Path(folder)
    if not folder.is_dir():
        print(f"Error: {folder} is not a valid directory")
        sys.exit(1)

    for file in folder.glob("*.csv"):
        try:
            # Read CSV
            df = pd.read_csv(file)

            # Take first 1/3 of rows
            subset_len = max(1, len(df) // 4)
            df_subset = df.iloc[:subset_len]

            # Save to new file
            out_file = file.with_name(file.stem + "_subset.csv")
            df_subset.to_csv(out_file, index=False)

            print(f"[INFO] Created {out_file}")

            # Run sanity_numbers.py on the new file
            result = subprocess.run(
                ["./sanity_numbers.py", str(out_file)],
                capture_output=True,
                text=True
            )
            print(f"[OUTPUT] {out_file.name}:\n{result.stdout}")
            if result.stderr:
                print(f"[ERROR OUTPUT] {result.stderr}")

        except Exception as e:
            print(f"[ERROR] Failed on {file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <folder_of_csvs>")
        sys.exit(1)

    process_csvs(sys.argv[1])
