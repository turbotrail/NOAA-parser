"""
dipole_evolution.py

Level 2: Dipole Evolution
Extracts dipole coefficients from IGRF-14, calculates dipole pole coordinates,
dipole moment strength, tilt angle, velocity, and acceleration.
Generates publication-quality plots and summary CSV datasets.
"""

import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points on Earth in km."""
    R = 6371.2 # Earth mean radius in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def fetch_dipole_coefficients(url):
    """Fetch and parse g10, g11, h11 from IGRF-14 file."""
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
            
    return np.array(years), np.array(g10), np.array(g11), np.array(h11)

def main():
    url = "https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt"
    print("Fetching IGRF-14 coefficients for Dipole Evolution analysis...")
    years, g10, g11, h11 = fetch_dipole_coefficients(url)
    
    # 1. Compute Dipole Strength (m) in nT
    m = np.sqrt(g10**2 + g11**2 + h11**2)
    
    # 2. Compute Dipole Tilt (offset from geographic axis in degrees)
    mh = np.sqrt(g11**2 + h11**2)
    tilt = np.degrees(np.arctan2(mh, -g10))
    
    # 3. Compute Lat/Lon of Dipole Pole
    lat = np.degrees(np.arctan2(-g10, mh))
    lon = np.degrees(np.arctan2(-h11, -g11))
    
    # 4. Compute Velocity and Acceleration
    velocities = np.zeros(len(years))
    accelerations = np.zeros(len(years))
    
    # Compute velocity (forward difference)
    for i in range(len(years) - 1):
        dist_km = haversine(lat[i], lon[i], lat[i+1], lon[i+1])
        dt = years[i+1] - years[i]
        velocities[i] = dist_km / dt
    
    # Last velocity is same as previous (boundary condition)
    velocities[-1] = velocities[-2]
    
    # Compute acceleration (forward difference of velocity)
    for i in range(len(years) - 1):
        dt = years[i+1] - years[i]
        accelerations[i] = (velocities[i+1] - velocities[i]) / dt
        
    accelerations[-1] = accelerations[-2]

    # Save to CSV
    out_dir = "outputs"
    os.makedirs(os.path.join(out_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    
    df = pd.DataFrame({
        "Year": years,
        "g10_nT": g10,
        "g11_nT": g11,
        "h11_nT": h11,
        "Dipole_Strength_nT": m,
        "Dipole_Tilt_deg": tilt,
        "Latitude_N": lat,
        "Longitude_E": lon,
        "Velocity_km_yr": velocities,
        "Acceleration_km_yr2": accelerations
    })
    
    csv_path = os.path.join(out_dir, "csv", "dipole_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    
    # Plotting
    print("Generating dipole evolution plots...")
    fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    
    # A. Dipole Strength
    axs[0].plot(years, m, 'o-', color='tab:blue', lw=2)
    axs[0].set_ylabel("Strength (nT)")
    axs[0].set_title("Dipole Moment Strength Evolution", fontweight='bold')
    axs[0].grid(True, ls='--')
    
    # B. Dipole Tilt
    axs[1].plot(years, tilt, 's-', color='tab:green', lw=2)
    axs[1].set_ylabel("Tilt Angle (°)")
    axs[1].set_title("Dipole Tilt relative to Geographic Axis", fontweight='bold')
    axs[1].grid(True, ls='--')
    
    # C. Pole Velocity
    axs[2].plot(years[:-1], velocities[:-1], 'd-', color='tab:red', lw=2) # omit last boundary point
    axs[2].set_ylabel("Velocity (km/yr)")
    axs[2].set_title("Dipole Pole Drift Velocity", fontweight='bold')
    axs[2].grid(True, ls='--')
    
    # D. Trajectory Latitude
    axs[3].plot(years, lat, '^-', color='tab:purple', lw=2)
    axs[3].set_ylabel("Latitude (°N)")
    axs[3].set_xlabel("Year")
    axs[3].set_title("Dipole Pole Latitude", fontweight='bold')
    axs[3].grid(True, ls='--')
    
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "figures", "dipole_evolution.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    print(f"Figure saved to {fig_path}")

if __name__ == "__main__":
    main()
