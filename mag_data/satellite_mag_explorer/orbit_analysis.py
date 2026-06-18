import pandas as pd
import numpy as np

def detect_saa_encounters(mag_df: pd.DataFrame, f_threshold: float = 25000.0) -> pd.DataFrame:
    """
    Detects periods where the satellite passes through the South Atlantic Anomaly.
    Uses a simple total field threshold (F < f_threshold).
    """
    is_saa = mag_df['F'] < f_threshold
    
    # Identify entry/exit points (transitions)
    saa_changes = is_saa.astype(int).diff().fillna(0)
    
    entries = mag_df[saa_changes == 1].index.tolist()
    exits = mag_df[saa_changes == -1].index.tolist()
    
    # Handle boundary conditions
    if is_saa.iloc[0]:
        entries.insert(0, mag_df.index[0])
    if is_saa.iloc[-1]:
        exits.append(mag_df.index[-1])
        
    encounters = []
    for entry, exit_idx in zip(entries, exits):
        segment = mag_df.loc[entry:exit_idx]
        if segment.empty:
            continue
            
        min_f_idx = segment['F'].idxmin()
        duration_s = (segment['Time'].iloc[-1] - segment['Time'].iloc[0]).total_seconds()
        
        encounters.append({
            'Entry_Time': segment['Time'].iloc[0],
            'Exit_Time': segment['Time'].iloc[-1],
            'Duration_s': duration_s,
            'Min_F_nT': segment['F'].min(),
            'Min_F_Lat': segment.loc[min_f_idx, 'Latitude'],
            'Min_F_Lon': segment.loc[min_f_idx, 'Longitude']
        })
        
    return pd.DataFrame(encounters)

def find_magnetic_equator_crossings(mag_df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds points where the satellite crosses the magnetic equator (Inclination changes sign).
    """
    # Detect sign change in Inclination
    inc_sign = np.sign(mag_df['Inc'])
    crossings = inc_sign.diff().fillna(0) != 0
    return mag_df[crossings].copy()
