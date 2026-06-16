import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Geomagnetic Secular Variation from USGS Observatory Data")
    parser.add_argument("input_csv", help="Path to the exported InfluxDB CSV file.")
    parser.add_argument("--output-dir", default="eda_outputs", help="Directory where plot outputs will be written.")
    parser.add_argument("--observatory", help="Comma-separated list of observatories to include.")
    return parser.parse_args()

def load_influx_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment="#", na_values=["", "null", "NULL"])
    df["_time"] = pd.to_datetime(df["_time"], utc=True, errors="coerce")
    df["_value"] = pd.to_numeric(df["_value"], errors="coerce")
    df = df.dropna(subset=["_time", "_value"]).reset_index(drop=True)
    return df

def clean_and_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the dataframe to have X, Y, Z, F as columns and remove obvious glitches."""
    pivot = df.pivot_table(index=["_time", "observatory"], columns="_field", values="_value").reset_index()
    
    components = [c for c in ['X', 'Y', 'Z', 'F'] if c in pivot.columns]
    
    # Sort to ensure consecutive diffs make sense
    pivot = pivot.sort_values(by=["observatory", "_time"])
    
    # Remove obvious instrument glitches: jumps > 5000 nT between consecutive samples
    for comp in components:
        diff = pivot.groupby("observatory")[comp].diff().abs()
        glitch_mask = diff > 5000
        pivot.loc[glitch_mask, comp] = np.nan
        
    return pivot

def compute_monthly_medians(pivot: pd.DataFrame) -> pd.DataFrame:
    """Group by month and compute median, then calculate decimal year."""
    components = [c for c in ['X', 'Y', 'Z', 'F'] if c in pivot.columns]
    
    # 1. Group data by month and compute median
    monthly = pivot.groupby(
        ['observatory', pd.Grouper(key='_time', freq='MS')]
    )[components].median().reset_index()
    
    # 2. Convert time to decimal years
    def to_decimal_year(ts):
        year = ts.year
        day_of_year = ts.dayofyear
        days_in_year = 366 if ts.is_leap_year else 365
        return year + (day_of_year - 1) / days_in_year

    monthly['year_decimal'] = monthly['_time'].apply(to_decimal_year)
    return monthly
    
from scipy.stats import theilslopes, linregress

def detect_step_change(series, threshold=100):
    diffs = series.diff().abs()
    return diffs.max() > threshold, diffs.max()

def fit_secular_variation(monthly: pd.DataFrame, output_dir: str):
    """Fit linear secular trends (OLS and Theil-Sen), validate F, and plot."""
    os.makedirs(output_dir, exist_ok=True)
    table_data = []
    
    observatories = monthly['observatory'].unique()
    
    for obs in observatories:
        obs_data = monthly[monthly['observatory'] == obs].copy()
        
        if len(obs_data) < 12:
            print(f"Skipping {obs}: Less than 12 monthly medians available.")
            continue
            
        # Compute F_calc
        if all(c in obs_data.columns for c in ['X', 'Y', 'Z', 'F']):
            obs_data['F_calc'] = np.sqrt(obs_data['X']**2 + obs_data['Y']**2 + obs_data['Z']**2)
            obs_data['F_residual'] = obs_data['F'] - obs_data['F_calc']
            f_res_max = obs_data['F_residual'].abs().max()
            f_res_mean = obs_data['F_residual'].abs().mean()
        else:
            f_res_max = np.nan
            f_res_mean = np.nan
            
        obs_sv = {"observatory": obs, "F_residual_max": f_res_max, "F_residual_mean": f_res_mean}
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        sv_vector_components = []
        
        print(f"\n--- Validation for {obs} ---")
        if not np.isnan(f_res_max):
            print(f"F vs F_calc Residual -> Max: {f_res_max:.2f} nT, Mean: {f_res_mean:.2f} nT")
            if f_res_max > 50:
                print(f"WARNING: Large inconsistency between F and (X,Y,Z) components for {obs}.")
        
        for idx, comp in enumerate(['X', 'Y', 'Z', 'F']):
            ax = axes[idx]
            if comp not in obs_data.columns:
                ax.set_visible(False)
                continue
                
            valid_data = obs_data.dropna(subset=[comp, 'year_decimal'])
            if len(valid_data) < 12:
                ax.set_title(f"{comp} Component (Not enough valid data)")
                continue
                
            t = valid_data['year_decimal']
            y = valid_data[comp]
            
            # OLS
            slope, intercept, r_value, p_value, stderr = linregress(t, y)
            
            # Theil-Sen
            ts_slope, ts_intercept, ts_low_slope, ts_high_slope = theilslopes(y, t)
            
            # Checks
            has_step, max_step = detect_step_change(y, threshold=50)
            if has_step:
                print(f"WARNING: Potential step change detected in {obs} {comp} (Max diff: {max_step:.1f} nT)")
            
            # Flag OLS vs Theil-Sen
            diff_pct = abs((slope - ts_slope) / (slope if slope != 0 else 1e-9)) * 100
            flag_ts = diff_pct > 20
            if flag_ts:
                print(f"FLAG: {obs} {comp} - OLS ({slope:.2f}) and Theil-Sen ({ts_slope:.2f}) differ by {diff_pct:.1f}%")
            
            # Store results
            obs_sv[f"{comp}_ols_slope"] = slope
            obs_sv[f"{comp}_ts_slope"] = ts_slope
            obs_sv[f"{comp}_stderr"] = stderr
            obs_sv[f"{comp}_R2"] = r_value**2
            
            if comp in ['X', 'Y', 'Z']:
                sv_vector_components.append(ts_slope**2)  # Using robust slope
            
            # Visualization
            ax.plot(t, y, 'o', label='Monthly Medians', color='tab:blue')
            ax.plot(t, intercept + slope * t, '--', color='gray', label=f'OLS: {slope:+.2f} nT/yr')
            ax.plot(t, ts_intercept + ts_slope * t, '-', color='tab:red', label=f'Theil-Sen: {ts_slope:+.2f} nT/yr')
            
            flag_text = "\n[!] Slopes differ >20%" if flag_ts else ""
            ax.text(0.05, 0.95, f"OLS SV = {slope:+.2f} ± {stderr:.2f} nT/yr\nTS SV = {ts_slope:+.2f} nT/yr{flag_text}", 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f"{obs} - {comp} Component")
            ax.set_xlabel("Decimal Year")
            ax.set_ylabel(f"{comp} (nT)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right')
            
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"secular_variation_{obs}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        
        # Calculate total secular variation magnitude using TS
        if len(sv_vector_components) == 3:
            obs_sv["Vector_SV_Magnitude_TS"] = np.sqrt(sum(sv_vector_components))
        else:
            obs_sv["Vector_SV_Magnitude_TS"] = np.nan
            
        table_data.append(obs_sv)
        
    # Generate Deliverables
    if table_data:
        df_results = pd.DataFrame(table_data)
        csv_path = os.path.join(output_dir, "secular_variation_summary.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"\nSaved validation and summary CSV to {csv_path}")
        
        print("\n" + "="*50)
        print("    Robust Secular Variation Report (nT/year)")
        print("="*50)
        for row in table_data:
            obs = row['observatory']
            print(f"\nObservatory: {obs}")
            print("-" * 25)
            for comp in ['X', 'Y', 'Z', 'F']:
                if f"{comp}_ts_slope" in row:
                    flag = " [!]" if f"{comp}_ols_slope" in row and abs(row[f"{comp}_ols_slope"] - row[f"{comp}_ts_slope"]) / max(1e-9, abs(row[f"{comp}_ols_slope"])) > 0.2 else ""
                    print(f"  d{comp}/dt (TS):  {row[f'{comp}_ts_slope']:+8.2f} nT/year {flag}")
                    print(f"  d{comp}/dt (OLS): {row[f'{comp}_ols_slope']:+8.2f} ± {row[f'{comp}_stderr']:5.2f} nT/year")
            
            if not np.isnan(row.get('Vector_SV_Magnitude_TS', np.nan)):
                print(f"\n  Vector SV Magnitude (TS): {row['Vector_SV_Magnitude_TS']:.2f} nT/year")
    else:
        print("No valid data available to compute Secular Variation.")

def main():
    args = parse_args()
    print("Loading InfluxDB data...")
    df = load_influx_csv(args.input_csv)
    
    if args.observatory:
        obs_list = [o.strip() for o in args.observatory.split(",")]
        df = df[df["observatory"].astype(str).isin(obs_list)]
        
    print("Pivoting and applying quality control (removing glitches > 5000 nT)...")
    pivot = clean_and_pivot(df)
    
    print("Computing monthly medians & decimal years...")
    monthly = compute_monthly_medians(pivot)
    
    print("Fitting linear secular trends...")
    fit_secular_variation(monthly, args.output_dir)
    print("\nDone.")

if __name__ == "__main__":
    main()
