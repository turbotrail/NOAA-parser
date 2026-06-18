import pandas as pd
import pyIGRF

def compute_magnetic_field(ephemeris_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes IGRF-14 field values for each point in the ephemeris.
    Expects DataFrame with 'Time', 'Latitude', 'Longitude', 'Altitude_km'.
    """
    results = []
    for _, row in ephemeris_df.iterrows():
        time_val = row['Time']
        # Convert datetime to fractional year for IGRF
        year = time_val.year + time_val.timetuple().tm_yday / 365.25
        
        # pyIGRF.igrf_value returns:
        # D (declination), I (inclination), H (horizontal), X (North), Y (East), Z (Vertical), F (Total)
        d, i, h, x, y, z, f = pyIGRF.igrf_value(
            lat=row['Latitude'], 
            lon=row['Longitude'], 
            alt=row['Altitude_km'], 
            year=year
        )
        results.append({
            'Dec': d, 'Inc': i, 'H': h, 
            'X': x, 'Y': y, 'Z': z, 'F': f
        })
        
    mag_df = pd.DataFrame(results)
    
    # Concatenate columns
    return pd.concat([ephemeris_df.reset_index(drop=True), mag_df.reset_index(drop=True)], axis=1)
