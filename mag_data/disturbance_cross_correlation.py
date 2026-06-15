import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute disturbance field (dF = F - daily median) and calculate cross-correlation with time lag."
    )
    parser.add_argument("input_csv", help="Path to the exported InfluxDB CSV file.")
    parser.add_argument("--output-dir", default="eda_outputs", help="Directory where plot outputs will be written.")
    parser.add_argument("--field", default="F", help="Specific _field to compute for (defaults to F).")
    parser.add_argument("--observatory", help="Comma-separated list of observatories to include.")
    parser.add_argument("--start-date", help="Filter data on or after this date (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Filter data on or before this date (YYYY-MM-DD).")
    parser.add_argument("--resample", default="1h", help="Resample frequency (e.g., 1h, 15min). Essential for time lag calculation.")
    parser.add_argument("--max-lag", type=int, default=24, help="Maximum number of lag steps (based on resample frequency) to evaluate.")
    parser.add_argument("--method", choices=["daily_median", "derivative"], default="daily_median", help="Method to calculate disturbance. 'derivative' is better for storm events.")
    return parser.parse_args()

def load_influx_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#", na_values=["", "null", "NULL"])
    df["_time"] = pd.to_datetime(df["_time"], utc=True, errors="coerce")
    df["_value"] = pd.to_numeric(df["_value"], errors="coerce")
    df = df.dropna(subset=["_time", "_value"]).reset_index(drop=True)
    return df

def filter_data(df: pd.DataFrame, field: str, observatory: str | None = None, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    df = df[df["_field"].astype(str) == field]
    if observatory:
        obs_list = [o.strip() for o in observatory.split(",")]
        df = df[df["observatory"].astype(str).isin(obs_list)]
    if start_date:
        df = df[df["_time"] >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        df = df[df["_time"] <= pd.to_datetime(end_date, utc=True)]
    return df.reset_index(drop=True)

def compute_ccf(s1: pd.Series, s2: pd.Series, max_lag: int):
    """Compute cross-correlation function for a range of lags."""
    lags = range(-max_lag, max_lag + 1)
    corrs = []
    for lag in lags:
        # If lag is positive, s2 is shifted forward (s2 is delayed relative to s1)
        # corr(s1, s2.shift(lag)) gives the correlation
        corr = s1.corr(s2.shift(lag))
        corrs.append(corr)
    return list(lags), corrs

def analyze_disturbance_and_xcorr(df: pd.DataFrame, output_dir: str, resample_freq: str, max_lag: int, field: str, method: str):
    os.makedirs(output_dir, exist_ok=True)
    
    if df["observatory"].nunique() < 2:
        print("Need at least 2 observatories to compute cross-correlation. Exiting.")
        return
        
    pivot = df.pivot_table(index="_time", columns="observatory", values="_value", aggfunc="mean")
    
    print(f"Resampling data at {resample_freq}...")
    pivot = pivot.resample(resample_freq.lower()).mean()
    
    if method == "daily_median":
        print("Computing daily medians and disturbance field...")
        daily_median = pivot.groupby(pivot.index.date).transform('median')
        dF = pivot - daily_median
        title_dF = f"Disturbance Field (d{field} = {field} - Daily Median)"
        ylabel_dF = f"d{field} Value"
    elif method == "derivative":
        print("Computing time derivative (dF/dt)...")
        dF = pivot.diff()
        title_dF = f"Time Derivative (d{field}/dt)"
        ylabel_dF = f"Change per {resample_freq}"
    else:
        raise ValueError("Unknown method")
        
    plt.figure(figsize=(14, 6))
    dF.plot(ax=plt.gca(), linewidth=1, alpha=0.8)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title(title_dF)
    plt.xlabel("Time")
    plt.ylabel(ylabel_dF)
    
    plt.legend(title="Observatory", loc='upper right')
    plt.tight_layout()
    df_plot_path = os.path.join(output_dir, f"disturbance_field_{field}.png")
    plt.savefig(df_plot_path, dpi=200)
    plt.close()
    print(f"Saved disturbance field plot to {df_plot_path}")
    
    # Calculate cross-correlation between all unique pairs
    print(f"Calculating cross-correlation with max lag = {max_lag} steps...")
    observatories = list(dF.columns)
    
    plt.figure(figsize=(10, 6))
    
    for i in range(len(observatories)):
        for j in range(i + 1, len(observatories)):
            obs1 = observatories[i]
            obs2 = observatories[j]
            
            lags, corrs = compute_ccf(dF[obs1], dF[obs2], max_lag)
            
            # Find the lag with maximum absolute correlation
            valid_corrs = [c for c in corrs if not np.isnan(c)]
            if valid_corrs:
                max_corr_idx = np.argmax(np.abs(valid_corrs))
                best_lag = lags[corrs.index(valid_corrs[max_corr_idx])]
                best_corr = valid_corrs[max_corr_idx]
                label = f"{obs1} vs {obs2} (Max Corr: {best_corr:.2f} at lag {best_lag})"
            else:
                label = f"{obs1} vs {obs2} (No valid data)"
                
            plt.plot(lags, corrs, marker='o', markersize=3, label=label)
            
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title(f"Cross-Correlation of Disturbance Field (d{field})")
    plt.xlabel(f"Time Lag (steps of {resample_freq})")
    plt.ylabel("Pearson Correlation")
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    ccf_plot_path = os.path.join(output_dir, f"cross_correlation_d{field}.png")
    plt.savefig(ccf_plot_path, dpi=200)
    plt.close()
    print(f"Saved cross-correlation plot to {ccf_plot_path}")
    print("-" * 40)

def main():
    args = parse_args()
    print("Loading data...")
    df = load_influx_csv(args.input_csv)
    df = filter_data(df, field=args.field, observatory=args.observatory, start_date=args.start_date, end_date=args.end_date)
    
    if df.empty:
        print("No data available for the given filters.")
        return
        
    analyze_disturbance_and_xcorr(df, args.output_dir, args.resample, args.max_lag, args.field, args.method)
    print("Done.")

if __name__ == "__main__":
    main()
