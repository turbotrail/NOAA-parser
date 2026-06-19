# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "scikit-learn", "numpy", "hapiclient", "pandas"]
# ///

import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
import warnings
import urllib.request
import json
from datetime import datetime, timedelta, timezone

def get_latest_f107():
    url = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
    try:
        print("Fetching latest F10.7 index from NOAA...")
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            data.sort(key=lambda x: x.get("time_tag", ""), reverse=True)
            flux = float(data[0].get("flux", 100.0))
            print(f"Successfully fetched F10.7: {flux}")
            return flux
    except Exception as e:
        print(f"Failed to fetch F10.7 from NOAA: {e}")
        print("Falling back to default average (150.0).")
        return 150.0

def get_latest_available_time():
    url = "https://vires.services/hapi/info?id=SW_OPER_IBIATMS_2F"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            stop_str = data.get("stopDate")
            if stop_str.endswith('Z'):
                stop_str = stop_str[:-1]
            return datetime.fromisoformat(stop_str).replace(tzinfo=timezone.utc)
    except Exception as e:
        return datetime.now(timezone.utc)

def get_historical_f107(date_str):
    try:
        start_date = datetime.strptime(date_str, '%Y-%m-%d')
        end_date = start_date + timedelta(days=1)
        end_str = end_date.strftime('%Y-%m-%d')
        url = f"https://lasp.colorado.edu/lisird/latis/dap/penticton_radio_flux.csv?time>={date_str}&time<={end_str}"
        print(f"Fetching historical F10.7 index for {date_str} from LISIRD...")
        with urllib.request.urlopen(url, timeout=10) as response:
            lines = response.read().decode().strip().split('\n')
            if len(lines) > 1:
                # Use the first recorded flux of the day
                flux = float(lines[1].split(',')[1])
                print(f"Successfully fetched historical F10.7: {flux}")
                return flux
            else:
                print(f"No F10.7 data found for {date_str}. Falling back to default (150.0).")
                return 150.0
    except Exception as e:
        print(f"Failed to fetch historical F10.7: {e}. Falling back to default (150.0).")
        return 150.0

class IBPNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 63),
            nn.BatchNorm1d(63),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2),
            nn.Linear(63, 63),
            nn.BatchNorm1d(63),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2),
            nn.Linear(63, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def load_model(model_path):
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        data = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
        
    scaler = data['scaler']
    state_dict = data['model_state_dict']
    
    model = IBPNetwork()
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler

def predict_prob(model, scaler, lt, doy, lon, f107):
    doy_sin = np.sin(doy / 365.25 * 2 * np.pi)
    doy_cos = np.cos(doy / 365.25 * 2 * np.pi)
    lt_sin = np.sin(lt / 24.0 * 2 * np.pi)
    lt_cos = np.cos(lt / 24.0 * 2 * np.pi)
    lon_sin = np.sin(lon / 360.0 * 2 * np.pi)
    lon_cos = np.cos(lon / 360.0 * 2 * np.pi)
    
    f107_scaled = scaler.transform(np.array([[f107]], dtype=np.float32))[0, 0]
    features_scaled = np.array([[doy_sin, doy_cos, lt_sin, lt_cos, lon_sin, lon_cos, f107_scaled]], dtype=np.float32)
    
    with torch.no_grad():
        x = torch.tensor(features_scaled, dtype=torch.float32)
        logits = model(x)
        return torch.sigmoid(logits).item()

def run_verification(model, scaler, f107, verify_date=None):
    from hapiclient import hapi
    import pandas as pd
    
    if verify_date:
        try:
            start_time = datetime.strptime(verify_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
            end_time = start_time + timedelta(days=1)
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
            return
    else:
        latest_time = get_latest_available_time()
        now = datetime.now(timezone.utc)
        end_time = min(latest_time, now)
        start_time = end_time - timedelta(days=1)
    
    print(f"\nFetching Swarm tracks for verification ({start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')})...")
    
    datasets = {
        'Swarm A': 'SW_OPER_IBIATMS_2F',
        'Swarm B': 'SW_OPER_IBIBTMS_2F',
        'Swarm C': 'SW_OPER_IBICTMS_2F'
    }
    
    all_data = []
    
    for sat_name, dataset in datasets.items():
        try:
            data, meta = hapi('https://vires.services/hapi', dataset, 'Latitude,Longitude,Bubble_Index', 
                              start_time.strftime('%Y-%m-%dT%H:%M:%SZ'), 
                              end_time.strftime('%Y-%m-%dT%H:%M:%SZ'))
            df = pd.DataFrame(data)
            if 'Bubble_Index' in df.columns:
                df['Satellite'] = sat_name
                all_data.append(df)
        except Exception as e:
            print(f"Warning: Failed to fetch verification data for {sat_name} ({e})")

    if not all_data:
        print("Could not find Bubble_Index data to verify from any satellite.")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    df = combined_df
        
    # Decode timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x))
    
    # Calculate Local Time and DOY
    utc_hours = df['Timestamp'].dt.hour + df['Timestamp'].dt.minute / 60.0 + df['Timestamp'].dt.second / 3600.0
    df['LT'] = (utc_hours + df['Longitude'] / 15.0) % 24.0
    df['DOY'] = df['Timestamp'].dt.dayofyear
    
    # Filter for Evening Equatorial regions (where bubbles happen)
    # LT between 18:00 and 02:00, Latitude near equator
    mask = (abs(df['Latitude']) < 15.0) & ((df['LT'] > 18.0) | (df['LT'] < 2.0))
    df = df[mask]
    
    bubbles = df[df['Bubble_Index'] == 1].copy()
    non_bubbles = df[df['Bubble_Index'] == 0].copy()
    
    # Sample up to 5 actual bubbles and 5 actual non-bubbles
    sample_bubbles = bubbles.sample(min(5, len(bubbles))) if len(bubbles) > 0 else pd.DataFrame()
    sample_non = non_bubbles.sample(min(5, len(non_bubbles))) if len(non_bubbles) > 0 else pd.DataFrame()
    
    test_cases = pd.concat([sample_bubbles, sample_non])
    if len(test_cases) == 0:
        print("No valid equatorial evening passes found in this timeframe.")
        return

    print("\n========================= VERIFICATION MODE =========================")
    print(f"{'Satellite':<9} | {'Time (UTC)':<16} | {'LT':<5} | {'Lon':<6} | {'Actual Bubble?':<15} | {'Predicted Prob'}")
    print("-" * 80)
    
    for _, row in test_cases.iterrows():
        prob = predict_prob(model, scaler, row['LT'], row['DOY'], row['Longitude'], f107)
        actual = "YES (1)" if row['Bubble_Index'] == 1 else "NO (0)"
        t_str = row['Timestamp'].strftime('%Y-%m-%d %H:%M')
        print(f"{row['Satellite']:<9} | {t_str:<16} | {row['LT']:5.1f} | {row['Longitude']:6.1f} | {actual:<15} | {prob*100:6.2f}%")
    print("=====================================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Swarm IBP Climatological Model Predictor")
    parser.add_argument("--lt", type=float, help="Local Time (0-24 hours)")
    parser.add_argument("--doy", type=float, help="Day of Year (1-365)")
    parser.add_argument("--lon", type=float, help="Longitude (-180 to 180 degrees)")
    parser.add_argument("--f107", type=float, default=None, help="F10.7 Solar Activity Index (auto-fetched if omitted)")
    parser.add_argument("--model", type=str, default="SW_OPER_IBP_CLI_2__00000000T000000_99999999T999999_0101.pth", help="Path to the .pth model file")
    parser.add_argument("--verify-latest", action="store_true", help="Run verification against the latest real-world Swarm A data")
    parser.add_argument("--verify-date", type=str, default=None, help="Run verification for a specific date (YYYY-MM-DD)")
    args = parser.parse_args()

    if not args.verify_latest and args.verify_date is None and (args.lt is None or args.doy is None or args.lon is None):
        parser.error("You must either provide --verify-latest, --verify-date, OR provide --lt, --doy, and --lon")

    if args.f107 is not None:
        f107_val = args.f107
    else:
        if args.verify_date:
            f107_val = get_historical_f107(args.verify_date)
        else:
            f107_val = get_latest_f107()

    model, scaler = load_model(args.model)

    if args.verify_latest or args.verify_date:
        run_verification(model, scaler, f107_val, args.verify_date)
    else:
        prob = predict_prob(model, scaler, args.lt, args.doy, args.lon, f107_val)
        print("\n" + "="*30)
        print("--- Input Parameters ---")
        print(f"Local Time:  {args.lt:5.1f} h")
        print(f"Day of Year: {args.doy:5.0f}")
        print(f"Longitude:   {args.lon:5.1f}°")
        print(f"F10.7 Index: {f107_val:5.1f}")
        print("-"*30)
        print(f"Predicted Bubble Probability: {prob * 100:.2f}%")
        print("="*30 + "\n")

if __name__ == "__main__":
    main()
