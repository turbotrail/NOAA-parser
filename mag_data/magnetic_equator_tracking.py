import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pyIGRF
import os

def main():
    years = np.arange(1900, 2026, 1)
    
    lons = np.arange(-180, 180.1, 2.0)
    lats = np.arange(-45, 45.1, 0.5) # Magnetic equator is well within +/- 45 degrees
    
    # We will store the equator latitudes for each year
    equator_lats = np.zeros((len(years), len(lons)))
    
    print("Computing IGRF values for magnetic equator...")
    for t, year in enumerate(years):
        for j, lon in enumerate(lons):
            I_vals = []
            for lat in lats:
                # pyIGRF.igrf_value returns (D, I, H, X, Y, Z, F)
                # I is at index 1
                try:
                    res = pyIGRF.igrf_value(lat, lon, 0, year)
                    I_vals.append(res[1])
                except Exception:
                    I_vals.append(np.nan)
            
            I_vals = np.array(I_vals)
            
            # Find where I crosses 0
            crossings = np.where(np.diff(np.sign(I_vals)))[0]
            if len(crossings) > 0:
                idx = crossings[0]
                # Linear interpolation for exact 0 crossing
                lat1, lat2 = lats[idx], lats[idx+1]
                i1, i2 = I_vals[idx], I_vals[idx+1]
                if i2 != i1:
                    zero_lat = lat1 - i1 * (lat2 - lat1) / (i2 - i1)
                else:
                    zero_lat = lat1
                equator_lats[t, j] = zero_lat
            else:
                equator_lats[t, j] = np.nan
        
        if year % 10 == 0:
            print(f"Processed year {year}")
            
    # Save to CSV
    os.makedirs("outputs/csv", exist_ok=True)
    df_dict = {"Longitude": lons}
    for t, year in enumerate(years):
        df_dict[f"Lat_{year}"] = equator_lats[t, :]
    df = pd.DataFrame(df_dict)
    df.to_csv("outputs/csv/magnetic_equator_tracking.csv", index=False)
    print("Metrics saved to outputs/csv/magnetic_equator_tracking.csv")
    
    # Plotting static map
    os.makedirs("outputs/figures", exist_ok=True)
    print("Generating static map...")
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor='#E0E0E0')
    ax.add_feature(cfeature.OCEAN, facecolor='#FFFFFF')
    ax.coastlines(linewidth=0.5)
    ax.gridlines(draw_labels=True, linestyle=':', color='gray', alpha=0.5)
    
    # In newer matplotlib, cm.get_cmap is deprecated in favor of matplotlib.colormaps
    try:
        cmap = plt.colormaps['viridis']
    except AttributeError:
        cmap = cm.get_cmap('viridis')
        
    norm = mcolors.Normalize(vmin=years[0], vmax=years[-1])
    
    for t, year in enumerate(years):
        if year % 5 == 0: # plot every 5 years for clarity
            ax.plot(lons, equator_lats[t, :], transform=ccrs.PlateCarree(),
                    color=cmap(norm(year)), linewidth=1.5, alpha=0.8)
            
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, aspect=40)
    cbar.set_label('Year')
    
    plt.title("Magnetic Equator Drift (1900 - 2025)\nInclination (I) = 0°")
    ax.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())
    plt.savefig("outputs/figures/magnetic_equator_drift.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create animation
    print("Generating map animation...")
    fig_map = plt.figure(figsize=(12, 6))
    fig_map.subplots_adjust(top=0.85)
    ax_map = fig_map.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    def update(frame):
        ax_map.clear()
        
        ax_map.add_feature(cfeature.LAND, facecolor='#E0E0E0')
        ax_map.add_feature(cfeature.OCEAN, facecolor='#FFFFFF')
        ax_map.coastlines(linewidth=0.5)
        ax_map.gridlines(draw_labels=True, linestyle=':', color='gray', alpha=0.5)
        
        # Plot historical lines in light gray
        for t in range(0, frame, 5):
            ax_map.plot(lons, equator_lats[t, :], transform=ccrs.PlateCarree(),
                        color='gray', linewidth=1.0, alpha=0.4)
                        
        # Plot current year
        year = years[frame]
        ax_map.plot(lons, equator_lats[frame, :], transform=ccrs.PlateCarree(),
                    color='red', linewidth=2.5)
                    
        ax_map.set_title(f"Magnetic Equator Drift (I = 0°)\nYear: {year}", pad=10)
        ax_map.set_extent([-180, 180, -30, 30], crs=ccrs.PlateCarree())

    ani = animation.FuncAnimation(fig_map, update, frames=len(years), interval=100, repeat_delay=2000)
    ani.save("outputs/figures/magnetic_equator_animation.gif", writer='pillow', fps=10, dpi=120)
    print("Animation saved to outputs/figures/magnetic_equator_animation.gif")
    
    # Calculate drift
    drift = np.abs(equator_lats[-1, :] - equator_lats[0, :])
    max_drift_idx = np.nanargmax(drift)
    max_drift_lon = lons[max_drift_idx]
    max_drift_val = drift[max_drift_idx]
    print(f"\nAnalysis complete.")
    print(f"Maximum drift observed at Longitude {max_drift_lon}°: {max_drift_val:.2f}° latitude change")

if __name__ == "__main__":
    main()
