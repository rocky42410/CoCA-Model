import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot anomaly flags from CSV with optional thresholding.")
    parser.add_argument("csv", help="Path to the CSV file")
    parser.add_argument("--thresh", type=float, default=None,
                        help="Optional threshold for marking anomalies (score >= thresh => 1). "
                             "If omitted, uses the 'label' column from CSV.")
    parser.add_argument("-o", "--output", help="Output PNG file (optional)")
    args = parser.parse_args()

    # Load the CSV
    df = pd.read_csv(args.csv)

    # Use threshold if given, otherwise fallback to label column
    if args.thresh is not None:
        df["anom_flag"] = (df["score"] >= args.thresh).astype(int)
        flag_source = f"Threshold = {args.thresh}"
    else:
        df["anom_flag"] = df["label"]
        flag_source = "CSV Labels"

    # Plot anomaly flag
    plt.figure(figsize=(10, 5))
    plt.plot(df["window_start_idx"], df["anom_flag"], "ro-", label="Anomaly Flag")

    # Labels and title
    plt.xlabel("Window Start Index")
    plt.ylabel("Anomaly Flag (0 = normal, 1 = anomaly)")
    plt.title(f"Anomaly Flags vs Window Index ({flag_source})")
    plt.yticks([0, 1])
    plt.grid(True)
    plt.legend()

    # Save or show
    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":
    main()
