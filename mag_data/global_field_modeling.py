"""
global_field_modeling.py

Level 3: Full Geomagnetic Field Modeling
Uses the pyIGRF library to generate global maps of Declination, Inclination, 
and Total Intensity for specific epochs (1900, 1950, 2000, 2025).
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyIGRF
import os

def compute_global_grid(year, resolution=5.0):
    """
    Compute IGRF values over a global grid.
    Returns lons, lats, D, I, F
    """
    lons = np.arange(-180, 180 + resolution, resolution)
    lats = np.arange(-90, 90 + resolution, resolution)
    
    Lon, Lat = np.meshgrid(lons, lats)
    D = np.zeros_like(Lon)
    I = np.zeros_like(Lon)
    F = np.zeros_like(Lon)
    H = np.zeros_like(Lon)
    
    for i in range(len(lats)):
        for j in range(len(lons)):
            # pyIGRF signature: igrf_value(lat, lon, alt, year)
            # returns: D, I, H, X, Y, Z, F
            try:
                res = pyIGRF.igrf_value(lats[i], lons[j], 0, year)
                D[i, j] = res[0]
                I[i, j] = res[1]
                H[i, j] = res[2]
                F[i, j] = res[6]
            except Exception as e:
                # Handle possible edge case singularities
                D[i, j] = np.nan
                I[i, j] = np.nan
                H[i, j] = np.nan
                F[i, j] = np.nan
                
    return Lon, Lat, D, I, F, H

def plot_global_map(Lon, Lat, data, title, cmap, clabel, out_path, is_declination=False):
    """Generate a global map using Cartopy."""
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    
    ax.add_feature(cfeature.LAND, facecolor='#E0E0E0')
    ax.add_feature(cfeature.OCEAN, facecolor='#FFFFFF')
    ax.coastlines(linewidth=0.5)
    
    # Levels for contour
    if is_declination:
        levels = np.arange(-180, 181, 10)
    else:
        levels = 20
        
    cf = ax.contourf(Lon, Lat, data, transform=ccrs.PlateCarree(),
                     cmap=cmap, levels=levels, extend='both', alpha=0.8)
    
    # Add contour lines
    cs = ax.contour(Lon, Lat, data, transform=ccrs.PlateCarree(),
                    colors='k', linewidths=0.5, levels=levels)
                    
    # Only label some contours to avoid clutter
    if is_declination:
        # Emphasize the agonic line (D=0)
        cs_zero = ax.contour(Lon, Lat, data, transform=ccrs.PlateCarree(),
                             colors='red', linewidths=2, levels=[0])
        ax.clabel(cs_zero, inline=True, fmt='%1.0f', fontsize=10)
    
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
    cbar.set_label(clabel)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    epochs = [1900, 1950, 2000, 2025]
    out_dir = os.path.join("outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    
    for year in epochs:
        print(f"Computing global grid for {year}...")
        Lon, Lat, D, I, F, H = compute_global_grid(year, resolution=5.0)
        
        # 1. Declination
        out_D = os.path.join(out_dir, f"global_declination_{year}.png")
        plot_global_map(Lon, Lat, D, 
                        f"Magnetic Declination ({year})", 
                        'twilight_shifted', 'Declination (Degrees East)', 
                        out_D, is_declination=True)
                        
        # 2. Inclination
        out_I = os.path.join(out_dir, f"global_inclination_{year}.png")
        plot_global_map(Lon, Lat, I, 
                        f"Magnetic Inclination ({year})", 
                        'coolwarm', 'Inclination (Degrees Down)', 
                        out_I)
                        
        # 3. Total Intensity
        out_F = os.path.join(out_dir, f"global_intensity_{year}.png")
        plot_global_map(Lon, Lat, F, 
                        f"Total Magnetic Intensity ({year})", 
                        'viridis', 'Total Intensity F (nT)', 
                        out_F)
                        
        # 4. Horizontal Intensity
        out_H = os.path.join(out_dir, f"global_horizontal_{year}.png")
        plot_global_map(Lon, Lat, H, 
                        f"Horizontal Magnetic Intensity ({year})", 
                        'plasma', 'Horizontal Intensity H (nT)', 
                        out_H)
                        
    print("Global maps generated successfully.")

if __name__ == "__main__":
    main()
