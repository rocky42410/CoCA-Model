#!/usr/bin/env python3
import pandas as pd
import sys

def main(path):
    df = pd.read_csv(path)
    if "joint_val" not in df.columns:
        raise ValueError("CSV has no column named 'joint_val'")
    col = df["joint_val"].dropna().astype(float)

    print(f"File: {path}")
    print(f"Count: {len(col)}")
    print(f"Max   : {col.max()}")
    print(f"Median: {col.median()}")
    print(f"Mean  : {col.mean()}")
    print(f"Min   : {col.min()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} data.csv")
        sys.exit(1)
    main(sys.argv[1])
