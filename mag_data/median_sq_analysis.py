import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze robust diurnal (Median Sq) patterns in geomagnetic data."
    )
    parser.add_argument("input_csv", help="Path to the exported InfluxDB CSV file.")
    parser.add_argument("--output-dir", default="eda_outputs", help="Directory where plot outputs will be written.")
    parser.add_argument("--field", default="F", help="Specific _field to compute for (defaults to F).")
    parser.add_argument("--observatory", help="Comma-separated list of observatories to include.")
    parser.add_argument("--start-date", help="Filter data on or after this date (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Filter data on or before this date (YYYY-MM-DD).")
    return parser.parse_args()

def load_influx_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#", na_values=["", "null", "NULL"])
    df["_time"] = pd.to_datetime(df["_time"], utc=True, errors="coerce")
    df["_value"] = pd.to_numeric(df["_value"], errors="coerce")
    df = df.dropna(subset=["_time", "_value"]).reset_index(drop=True)
    return df

def filter_data(df: pd.DataFrame, field: str, observatory: str | None = None, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    df = df[df["_field"].astype(str) == field].copy()
    if observatory:
        obs_list = [o.strip() for o in observatory.split(",")]
        df = df[df["observatory"].astype(str).isin(obs_list)]
    if start_date:
        df = df[df["_time"] >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        df = df[df["_time"] <= pd.to_datetime(end_date, utc=True)]
    return df.reset_index(drop=True)

def analyze_median_sq_pattern(df: pd.DataFrame, output_dir: str, field: str):
    os.makedirs(output_dir, exist_ok=True)
    
    if df.empty:
        print("No data available to analyze.")
        return

    print("Computing daily means to isolate diurnal variations...")
    # Calculate date and hour of day
    df['date'] = df['_time'].dt.date
    df['hour'] = df['_time'].dt.hour
    
    # Calculate daily mean for each observatory to remove baseline drift
    df['daily_mean'] = df.groupby(['date', 'observatory'])['_value'].transform('mean')
    
    # Calculate variation from daily mean
    df['variation'] = df['_value'] - df['daily_mean']
    
    print("Aggregating by hour of day (UTC)...")
    # Group by observatory and hour, calculating median, 25th, and 75th percentiles
    diurnal_stats = df.groupby(['observatory', 'hour'])['variation'].agg(
        median='median',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)
    ).reset_index()
    
    diurnal_stats['iqr'] = diurnal_stats['q75'] - diurnal_stats['q25']

    # Plotting
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 10), sharex=True)
    
    observatories = diurnal_stats['observatory'].unique()
    for obs in observatories:
        obs_data = diurnal_stats[diurnal_stats['observatory'] == obs].sort_values('hour')
        
        # Plot median variation
        ax1.plot(obs_data['hour'], obs_data['median'], marker='o', label=obs)
        
        # Plot IQR
        ax2.plot(obs_data['hour'], obs_data['iqr'], marker='s', label=obs)

    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_title(f"Robust Diurnal Pattern (Median Sq Variation) for Field: {field}")
    ax1.set_ylabel(f"Median Deviation ({field})")
    ax1.legend(title="Observatory", loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title(f"Interquartile Range (IQR) of Sq Variation for Field: {field}")
    ax2.set_xlabel("Hour of Day (UTC)")
    ax2.set_ylabel(f"IQR ({field})")
    ax2.set_xticks(range(0, 24))
    ax2.legend(title="Observatory", loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"median_sq_pattern_{field}.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    print(f"Saved robust diurnal pattern plot to {plot_path}")
    print("-" * 40)

def main():
    args = parse_args()
    print("Loading data...")
    df = load_influx_csv(args.input_csv)
    df = filter_data(df, field=args.field, observatory=args.observatory, start_date=args.start_date, end_date=args.end_date)
    
    analyze_median_sq_pattern(df, args.output_dir, args.field)
    print("Done.")

if __name__ == "__main__":
    main()
