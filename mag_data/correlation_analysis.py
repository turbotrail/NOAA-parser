import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    sns = None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and plot the correlation between stations."
    )
    parser.add_argument("input_csv", help="Path to the exported InfluxDB CSV file.")
    parser.add_argument("--output-dir", default="eda_outputs", help="Directory where plot outputs will be written.")
    parser.add_argument("--field", help="Specific _field to compute correlation for (e.g., F). If not provided, computes for all fields.")
    parser.add_argument("--observatory", help="Comma-separated list of observatories to include.")
    parser.add_argument("--start-date", help="Filter data on or after this date (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Filter data on or before this date (YYYY-MM-DD).")
    parser.add_argument("--resample", default="1h", help="Resample frequency before correlation (e.g., 1h, 1d). Important for aligning timestamps. Default is 1h.")
    return parser.parse_args()

def load_influx_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#", na_values=["", "null", "NULL"])
    df["_time"] = pd.to_datetime(df["_time"], utc=True, errors="coerce")
    df["_value"] = pd.to_numeric(df["_value"], errors="coerce")
    df = df.dropna(subset=["_time", "_value"]).reset_index(drop=True)
    return df

def filter_data(df: pd.DataFrame, field: str | None = None, observatory: str | None = None, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    if field:
        df = df[df["_field"].astype(str) == field]
    if observatory:
        obs_list = [o.strip() for o in observatory.split(",")]
        df = df[df["observatory"].astype(str).isin(obs_list)]
    if start_date:
        df = df[df["_time"] >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        df = df[df["_time"] <= pd.to_datetime(end_date, utc=True)]
    return df.reset_index(drop=True)

def plot_correlation(df: pd.DataFrame, output_dir: str, resample_freq: str):
    os.makedirs(output_dir, exist_ok=True)
    
    fields = df["_field"].unique()
    
    for f in fields:
        field_df = df[df["_field"] == f]
        if field_df.empty or field_df["observatory"].nunique() < 2:
            print(f"Skipping {f} because it doesn't have data for multiple observatories.")
            continue
            
        # Pivot so that each column is an observatory
        pivot = field_df.pivot_table(index="_time", columns="observatory", values="_value", aggfunc="mean")
        
        # Resampling is usually required because timestamps between different stations might not align exactly
        if resample_freq:
            pivot = pivot.resample(resample_freq.lower()).mean()
            
        corr_matrix = pivot.corr()
        
        plt.figure(figsize=(8, 6))
        if sns is not None:
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True)
        else:
            cax = plt.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar(cax)
            plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
            plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
            # Add annotations
            for i in range(len(corr_matrix.index)):
                for j in range(len(corr_matrix.columns)):
                    plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black")
                    
        plt.title(f"Station Correlation for Field: {f}")
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, f"correlation_heatmap_{f}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        
        print(f"Saved correlation plot for {f} to {out_path}")
        print(f"Correlation matrix for {f}:")
        print(corr_matrix)
        print("-" * 40)

def main():
    args = parse_args()
    print("Loading data...")
    df = load_influx_csv(args.input_csv)
    df = filter_data(df, field=args.field, observatory=args.observatory, start_date=args.start_date, end_date=args.end_date)
    
    if df.empty:
        print("No data available for the given filters.")
        return
        
    print("Computing and plotting correlations...")
    plot_correlation(df, args.output_dir, args.resample)
    print("Done.")

if __name__ == "__main__":
    main()
