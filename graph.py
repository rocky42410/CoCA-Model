import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot anomaly flags from CSV.")
    parser.add_argument("csv", help="Path to the CSV file")
    parser.add_argument("-o", "--output", help="Output PNG file (optional)")
    args = parser.parse_args()

    # Load the CSV
    df = pd.read_csv(args.csv)

    # Plot only the anomaly flag
    plt.figure(figsize=(10, 5))
    plt.plot(df["window_start_idx"], df["label"], "ro-", label="Anomaly Flag")

    # Labels and title
    plt.xlabel("Window Start Index")
    plt.ylabel("Anomaly Flag (0 = normal, 1 = anomaly)")
    plt.title("Anomaly Flags vs Window Index")
    plt.yticks([0, 1])  # force axis to 0/1
    plt.grid(True)
    plt.legend()

    # Save or show
    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":
    main()
