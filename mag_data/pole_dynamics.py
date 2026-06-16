"""
pole_dynamics.py

Level 4: Pole Dynamics & Comparison
Computes and compares Geomagnetic Dipole Poles and Magnetic Dip Poles.
Calculates drift trajectories and great-circle separations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyIGRF
import os
import urllib.request

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.2 # km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def fetch_dipole_pole(url):
    """Fetch dipole pole coordinates for all epochs."""
    req = urllib.request.urlopen(url)
    lines = req.read().decode('utf-8').split('\n')
    
    years, g10, g11, h11 = [], [], [], []
    for line in lines:
        if line.startswith('g/h'):
            parts = line.split()
            years = [float(y) for y in parts[3:-1]]
        elif line.startswith('g  1  0'):
            g10 = [float(v) for v in line.split()[3:-1]]
        elif line.startswith('g  1  1'):
            g11 = [float(v) for v in line.split()[3:-1]]
        elif line.startswith('h  1  1'):
            h11 = [float(v) for v in line.split()[3:-1]]
            
    g10, g11, h11 = np.array(g10), np.array(g11), np.array(h11)
    mh = np.sqrt(g11**2 + h11**2)
    lat = np.degrees(np.arctan2(-g10, mh))
    lon = np.degrees(np.arctan2(-h11, -g11))
    
    return np.array(years), lat, lon

def find_dip_pole(year, hemisphere='N'):
    """
    Find the magnetic dip pole where Inclination is exactly +/- 90 degrees.
    This corresponds to horizontal intensity H = 0.
    """
    if hemisphere == 'N':
        lat_range = np.arange(60, 90.1, 0.5)
        sign = 1
    else:
        lat_range = np.arange(-90, -59.9, 0.5)
        sign = -1
        
    lon_range = np.arange(-180, 180, 1.0)
    
    # 1. Coarse search for minimum H
    min_h = float('inf')
    best_lat = 0
    best_lon = 0
    
    for lat in lat_range:
        for lon in lon_range:
            try:
                res = pyIGRF.igrf_value(lat, lon, 0, year)
                # res[2] is H (horizontal intensity)
                h_val = res[2] 
                if h_val < min_h:
                    min_h = h_val
                    best_lat = lat
                    best_lon = lon
            except Exception:
                pass
                
    # 2. Fine search around the best point
    fine_lat_range = np.arange(best_lat - 1.0, best_lat + 1.01, 0.05)
    fine_lon_range = np.arange(best_lon - 2.0, best_lon + 2.01, 0.1)
    
    for lat in fine_lat_range:
        for lon in fine_lon_range:
            # Handle pole wrapping safely
            if lat > 90: lat = 180 - lat; lon = lon + 180
            if lat < -90: lat = -180 - lat; lon = lon + 180
            if lon > 180: lon -= 360
            if lon < -180: lon += 360
                
            try:
                res = pyIGRF.igrf_value(lat, lon, 0, year)
                h_val = res[2]
                if h_val < min_h:
                    min_h = h_val
                    best_lat = lat
                    best_lon = lon
            except Exception:
                pass
                
    return best_lat, best_lon

def plot_pole_comparison(years, dip_lat, dip_lon, dipole_lat, dipole_lon, out_path):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=-90))
    ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor='#E0E0E0', edgecolor='k', linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='#FFFFFF')
    ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
    
    # Plot trajectories
    ax.plot(dipole_lon, dipole_lat, 'o-', color='tab:blue', linewidth=2, markersize=5, 
            transform=ccrs.PlateCarree(), label='Geomagnetic Dipole Pole')
            
    ax.plot(dip_lon, dip_lat, 's-', color='tab:red', linewidth=2, markersize=5, 
            transform=ccrs.PlateCarree(), label='Magnetic Dip Pole')
            
    # Annotate years
    for i, year in enumerate(years):
        if year % 20 == 0 or i == 0 or i == len(years) - 1:
            ax.annotate(f"{int(year)}", (dipole_lon[i], dipole_lat[i]), 
                        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                        textcoords="offset points", xytext=(5, 5), color='blue', fontsize=8)
            ax.annotate(f"{int(year)}", (dip_lon[i], dip_lat[i]), 
                        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                        textcoords="offset points", xytext=(5, -10), color='red', fontsize=8)
                        
    ax.set_title("North Magnetic Poles Drift Comparison (1900-2025)", fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower left')
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    url = "https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt"
    print("Fetching dipole trajectories...")
    years, dipole_lat, dipole_lon = fetch_dipole_pole(url)
    
    dip_lat = np.zeros_like(years)
    dip_lon = np.zeros_like(years)
    separations = np.zeros_like(years)
    
    print("Calculating Magnetic Dip Poles (this requires grid search per epoch)...")
    for i, year in enumerate(years):
        print(f"Searching for {year}...")
        d_lat, d_lon = find_dip_pole(year, hemisphere='N')
        dip_lat[i] = d_lat
        dip_lon[i] = d_lon
        
        # Calculate great circle separation
        separations[i] = haversine(dip_lat[i], dip_lon[i], dipole_lat[i], dipole_lon[i])
        
    out_dir = "outputs"
    os.makedirs(os.path.join(out_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    
    df = pd.DataFrame({
        "Year": years,
        "Dipole_Pole_Lat_N": dipole_lat,
        "Dipole_Pole_Lon_E": dipole_lon,
        "Magnetic_Dip_Pole_Lat_N": dip_lat,
        "Magnetic_Dip_Pole_Lon_E": dip_lon,
        "Separation_Distance_km": separations
    })
    
    csv_path = os.path.join(out_dir, "csv", "pole_dynamics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    
    fig_path = os.path.join(out_dir, "figures", "pole_comparison_trajectory.png")
    plot_pole_comparison(years, dip_lat, dip_lon, dipole_lat, dipole_lon, fig_path)
    print(f"Trajectory plot saved to {fig_path}")

if __name__ == "__main__":
    main()
