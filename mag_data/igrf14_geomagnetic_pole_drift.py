"""
igrf14_geomagnetic_pole_drift.py

A scientifically accurate program to fetch IGRF-14 coefficients and visualize 
the Earth's geomagnetic dipole north pole drift from 1900 to 2025.
"""

import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

def fetch_igrf_coefficients(url):
    """
    Parse the IGRF-14 coefficient file.
    Extracts the degree-1 coefficients (g_1^0, g_1^1, h_1^1) representing
    the Earth's dipole field, and all available epochs.
    """
    req = urllib.request.urlopen(url)
    lines = req.read().decode('utf-8').split('\n')
    
    years, g10, g11, h11 = [], [], [], []
    for line in lines:
        if line.startswith('g/h'):
            parts = line.split()
            # Epochs start from the 4th column until the second to last (skipping SV column)
            years = [float(y) for y in parts[3:-1]]
        elif line.startswith('g  1  0'):
            g10 = [float(v) for v in line.split()[3:-1]]
        elif line.startswith('g  1  1'):
            g11 = [float(v) for v in line.split()[3:-1]]
        elif line.startswith('h  1  1'):
            h11 = [float(v) for v in line.split()[3:-1]]
            
    return np.array(years), np.array(g10), np.array(g11), np.array(h11)

def calculate_dipole_pole(g10, g11, h11):
    """
    Compute the geomagnetic dipole pole coordinates.
    The dipole moment vector m is defined from the degree-1 coefficients:
    m = (mx, my, mz) = (-g11, -h11, -g10)
    
    Geomagnetic pole latitude and longitude are calculated as:
    latitude = atan2(mz, sqrt(mx^2 + my^2))
    longitude = atan2(my, mx)
    """
    # Dipole moment vector components
    mx = -g11
    my = -h11
    mz = -g10
    
    # Calculate latitude and longitude in radians, then convert to degrees
    lat = np.degrees(np.arctan2(mz, np.sqrt(mx**2 + my**2)))
    lon = np.degrees(np.arctan2(my, mx))
    
    return lat, lon

def validate_coordinates(years, lat, lon):
    """
    Verify coordinates for 1900 and 2025 against expected values.
    1900 expected: ~78.6 N, 68.7 W (-68.7)
    2025 expected: ~80.8 N, 72.8 W (-72.8)
    """
    idx_1900 = np.where(years == 1900.0)[0]
    idx_2025 = np.where(years == 2025.0)[0]
    
    warnings = []
    
    if len(idx_1900) > 0:
        i = idx_1900[0]
        # Check against expected, using an absolute tolerance of 0.2 degrees
        if not (np.isclose(lat[i], 78.6, atol=0.2) and np.isclose(lon[i], -68.7, atol=0.2)):
            warnings.append(f"WARNING: 1900 coords {lat[i]:.2f}N, {lon[i]:.2f}E differ significantly from 78.6N, 68.7W")
            
    if len(idx_2025) > 0:
        i = idx_2025[0]
        if not (np.isclose(lat[i], 80.8, atol=0.2) and np.isclose(lon[i], -72.8, atol=0.2)):
            warnings.append(f"WARNING: 2025 coords {lat[i]:.2f}N, {lon[i]:.2f}E differ significantly from 80.8N, 72.8W")
            
    if warnings:
        for w in warnings:
            print(w)
    else:
        print("Validation checks passed: 1900 and 2025 coordinates match expected values.")

def print_verification_table(years, lat, lon):
    """Prints a clean tabular display of the computed pole locations."""
    print("\n--- Geomagnetic Dipole Pole Coordinates ---")
    print(f"{'Year':<10} | {'Latitude (°N)':<15} | {'Longitude (°E)':<15}")
    print("-" * 45)
    for y, la, lo in zip(years, lat, lon):
        print(f"{int(y):<10} | {la:<15.2f} | {lo:<15.2f}")
    print("-" * 45 + "\n")

def plot_static_trajectory(years, lat, lon, out_path):
    """
    Create a static trajectory plot of Latitude vs Longitude.
    """
    plt.figure(figsize=(10, 7))
    plt.plot(lon, lat, marker='o', linestyle='-', color='tab:red', markersize=6, linewidth=2, label="Pole Trajectory")
    
    # Label selected years (every 20 years + endpoints)
    for i, year in enumerate(years):
        if year % 20 == 0 or i == 0 or i == len(years) - 1:
            plt.annotate(f"{int(year)}", 
                         (lon[i], lat[i]), 
                         textcoords="offset points", 
                         xytext=(8, 8), 
                         ha='left',
                         fontsize=10,
                         fontweight='bold',
                         color='black')
            
    # Include clear title with disclaimer about magnetic dip pole
    plt.title("Geomagnetic North Pole Drift (1900 - 2025)\nNote: This is the geomagnetic dipole pole, NOT the magnetic dip pole", 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Longitude (°)", fontsize=12)
    plt.ylabel("Latitude (°N)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def animate_arctic_map(years, lat, lon, out_path):
    """
    Create an Arctic map animation using Cartopy, centered around the 
    trajectory of the dipole pole.
    """
    fig = plt.figure(figsize=(10, 10))
    # North Polar Stereographic projection centered near -70 longitude
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=-70))
    
    # Set extent to focus on the Arctic (72N to 90N)
    ax.set_extent([-180, 180, 72, 90], crs=ccrs.PlateCarree())
    
    # Add cartopy features for publication quality
    ax.add_feature(cfeature.OCEAN, facecolor='#E0F7FA')
    ax.add_feature(cfeature.LAND, facecolor='#EFEFEF', edgecolor='black')
    ax.coastlines(resolution='50m', linewidth=1.0)
    
    # Latitude and longitude gridlines
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_title("Geomagnetic Dipole North Pole Drift", fontsize=18, fontweight='bold', pad=20)
    
    # Plot elements to be updated in animation
    # Using transform=ccrs.PlateCarree() per requirements to map degrees properly
    line, = ax.plot([], [], 'o-', color='tab:red', markersize=4, linewidth=1.5, transform=ccrs.PlateCarree())
    
    year_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=18, 
                        fontweight='bold', verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
            
    def init():
        line.set_data([], [])
        year_text.set_text('')
        return line, year_text

    def update(frame):
        # We pass slicing up to frame+1 to show the cumulative trajectory drawing over time
        line.set_data(lon[:frame+1], lat[:frame+1])
        year_text.set_text(
            f"Year: {int(years[frame])}\n"
            f"{lat[frame]:.2f}°N\n"
            f"{abs(lon[frame]):.2f}°W"
        )
        return line, year_text

    print("Generating Cartopy Earth map GIF animation...")
    ani = FuncAnimation(fig, update, frames=len(years), init_func=init, blit=True, repeat=False)
    
    writer = PillowWriter(fps=5) 
    ani.save(out_path, writer=writer)
    plt.close()

def main():
    url = "https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt"
    print("1. Fetching IGRF-14 coefficients...")
    years, g10, g11, h11 = fetch_igrf_coefficients(url)
    
    print("2. Calculating Geomagnetic Dipole Pole trajectory...")
    lat, lon = calculate_dipole_pole(g10, g11, h11)
    
    print("3. Verification Table:")
    print_verification_table(years, lat, lon)
    
    print("4. Validating coordinates...")
    validate_coordinates(years, lat, lon)
    
    out_dir = "eda_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    static_plot_path = os.path.join(out_dir, "trajectory_plot.png")
    print("5. Generating static trajectory plot...")
    plot_static_trajectory(years, lat, lon, static_plot_path)
    
    anim_path = os.path.join(out_dir, "geomagnetic_pole_drift.gif")
    print("6. Generating Arctic map animation (this may take a moment)...")
    animate_arctic_map(years, lat, lon, anim_path)
    
    print("\nProcessing Complete!")
    print(f"Static plot saved to: {static_plot_path}")
    print(f"Animation saved to: {anim_path}")

if __name__ == "__main__":
    main()
