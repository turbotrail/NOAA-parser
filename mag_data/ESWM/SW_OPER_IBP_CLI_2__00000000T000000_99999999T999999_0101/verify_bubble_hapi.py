# /// script
# requires-python = ">=3.11"
# dependencies = ["hapiclient", "pandas", "matplotlib"]
# ///

from hapiclient import hapi
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime, timedelta, timezone

import urllib.request
import json

def get_latest_available_time():
    url = "https://vires.services/hapi/info?id=SW_OPER_IBIATMS_2F"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode())
            stop_str = data.get("stopDate")
            # Usually format is "YYYY-MM-DDTHH:MM:SSZ" or similar
            if stop_str.endswith('Z'):
                stop_str = stop_str[:-1]
            stop_time = datetime.fromisoformat(stop_str).replace(tzinfo=timezone.utc)
            return stop_time
    except Exception as e:
        print(f"Warning: Could not fetch latest available time from server ({e}).")
        return datetime.now(timezone.utc)

def main():
    latest_time = get_latest_available_time()
    
    # If the latest available time is in the future (unlikely), clamp to now. 
    # Otherwise use the latest available time as the end of our 24-hour window.
    now = datetime.now(timezone.utc)
    end_time = min(latest_time, now)
    
    default_stop = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    default_start = (end_time - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')

    parser = argparse.ArgumentParser(description="Verify Bubble Predictions using VirES HAPI Server")
    parser.add_argument("--start", type=str, default=default_start, help=f"Start time (default: {default_start})")
    parser.add_argument("--stop", type=str, default=default_stop, help=f"Stop time (default: {default_stop})")
    args = parser.parse_args()

    server = 'https://vires.services/hapi'
    datasets = {
        'Swarm A': 'SW_OPER_IBIATMS_2F',
        'Swarm B': 'SW_OPER_IBIBTMS_2F',
        'Swarm C': 'SW_OPER_IBICTMS_2F'
    }
    
    print(f"Fetching actual Swarm observations from {args.start} to {args.stop}...")
    
    all_data = []
    
    for sat_name, dataset in datasets.items():
        print(f"Querying {sat_name} ({dataset})...")
        try:
            data, meta = hapi(server, dataset, 'Latitude,Longitude,Bubble_Index', args.start, args.stop)
            df = pd.DataFrame(data)
            
            if 'Bubble_Index' in df.columns:
                df['Satellite'] = sat_name
                all_data.append(df)
            else:
                print(f"  Bubble_Index not found in {sat_name}.")
        except Exception as e:
            print(f"  Failed to fetch data for {sat_name}: {e}")

    if not all_data:
        print("No data fetched from any satellite.")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)
    bubbles = combined_df[combined_df['Bubble_Index'] == 1]
    
    print("\n--- Summary of Actual Observations ---")
    print(f"Total measurements across all satellites: {len(combined_df)}")
    print(f"Total positive bubble detections: {len(bubbles)}")
    
    if len(bubbles) > 0:
        print("\nLocations of observed bubbles:")
        for _, row in bubbles.head(5).iterrows():
            timestamp = row['Timestamp'].decode('utf-8') if isinstance(row['Timestamp'], bytes) else row['Timestamp']
            print(f"  [{row['Satellite']}] Time: {timestamp} | Lat: {row['Latitude']:5.2f} | Lon: {row['Longitude']:5.2f}")
        
        if len(bubbles) > 5:
            print(f"  ... and {len(bubbles) - 5} more.")
            
        plt.figure(figsize=(12, 6))
        
        # Plot full tracks in gray
        plt.scatter(combined_df['Longitude'], combined_df['Latitude'], c='whitesmoke', s=1, label='Satellite Tracks', zorder=1)
        
        colors = {'Swarm A': 'red', 'Swarm B': 'blue', 'Swarm C': 'green'}
        
        # Plot bubbles colored by satellite
        for sat_name, color in colors.items():
            sat_bubbles = bubbles[bubbles['Satellite'] == sat_name]
            if len(sat_bubbles) > 0:
                plt.scatter(sat_bubbles['Longitude'], sat_bubbles['Latitude'], c=color, s=15, label=f'{sat_name} Bubbles', zorder=2)
                
        plt.title(f"Swarm A, B & C Bubble Observations ({args.start[:10]})")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        filename = "actual_bubbles.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved as {filename}")
    else:
        print("No bubbles were observed during this timeframe.")

if __name__ == "__main__":
    main()
