import urllib.error
from skyfield.api import load, EarthSatellite, wgs84
import pandas as pd
from datetime import timedelta
import logging

def load_satellite_by_name(name, url='https://celestrak.org/NORAD/elements/stations.txt'):
    """
    Loads a satellite by its exact string name from Celestrak.
    """
    try:
        satellites = load.tle_file(url)
        by_name = {sat.name: sat for sat in satellites}
        if name not in by_name:
            raise ValueError(f"Satellite {name} not found in {url}")
        return by_name[name]
    except Exception as e:
        logging.warning(f"Could not load TLE from {url}: {e}. Falling back to hardcoded ISS TLE.")
        line1 = '1 25544U 98067A   23270.43577546  .00015504  00000+0  28114-3 0  9997'
        line2 = '2 25544  51.6416 313.3986 0003504 140.2311 319.4635 15.49845353417637'
        return load_satellite_from_tle("ISS (ZARYA)", line1, line2)

def load_satellite_from_tle(name, line1, line2):
    """
    Loads a satellite from user-provided TLE lines.
    """
    ts = load.timescale()
    return EarthSatellite(line1, line2, name, ts)

def generate_ephemeris(satellite, start_time, duration_hours=24, step_seconds=60):
    """
    Propagates the orbit and returns a DataFrame with coordinates.
    """
    ts = load.timescale()
    
    # Generate time steps
    num_steps = int(duration_hours * 3600 / step_seconds)
    times = [start_time + timedelta(seconds=i*step_seconds) for i in range(num_steps)]
    
    # Convert to skyfield time object
    t = ts.from_datetimes(times)
    
    # Compute position
    geocentric = satellite.at(t)
    subpoint = wgs84.subpoint(geocentric)
    
    df = pd.DataFrame({
        'Time': [ti.replace(tzinfo=None) for ti in times], # store naive datetime for pandas ease or keep aware
        'Latitude': subpoint.latitude.degrees,
        'Longitude': subpoint.longitude.degrees,
        'Altitude_km': subpoint.elevation.km
    })
    
    # Optional: ensure Time is datetime aware or UTC
    df['Time'] = pd.to_datetime(df['Time']).dt.tz_localize('UTC')
    return df
