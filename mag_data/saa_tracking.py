import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyIGRF
import os

def calculate_cell_area(lat, dlat, dlon):
    """
    Calculate approximate physical area of a grid cell in km^2.
    lat: center latitude of the cell in degrees
    dlat: grid resolution in degrees latitude
    dlon: grid resolution in degrees longitude
    R: Earth radius ~ 6371 km
    """
    R = 6371.0
    # convert to radians
    lat_rad = np.radians(lat)
    dlat_rad = np.radians(dlat)
    dlon_rad = np.radians(dlon)
    # the physical area approximation is reasonable for fine grids
    return (R**2) * np.cos(lat_rad) * dlat_rad * dlon_rad

def main():
    years = np.arange(1900, 2026, 1)
    res = 2.0
    lons = np.arange(-180, 180 + res, res)
    lats = np.arange(-90, 90 + res, res)
    Lon, Lat = np.meshgrid(lons, lats)
    
    # Pre-calculate cell areas
    cell_areas = np.zeros_like(Lat)
    for i in range(len(lats)):
        for j in range(len(lons)):
            cell_areas[i, j] = calculate_cell_area(lats[i], res, res)
            
    all_F = np.zeros((len(years), len(lats), len(lons)))
    
    metrics = {
        "Year": [],
        "Center_Lon": [],
        "Center_Lat": [],
        "Min_F_nT": [],
        "Area_km2": [],
        "Growth_Rate_km2_yr": []
    }
    
    threshold = 25000.0
    
    print("Computing IGRF values from 1900 to 2025...")
    for t, year in enumerate(years):
        min_F = np.inf
        min_loc = (0, 0)
        area = 0.0
        
        for i in range(len(lats)):
            for j in range(len(lons)):
                try:
                    res_val = pyIGRF.igrf_value(lats[i], lons[j], 0, year)
                    f_val = res_val[6]
                except Exception:
                    f_val = np.nan
                    
                all_F[t, i, j] = f_val
                
                if f_val < threshold:
                    area += cell_areas[i, j]
                
                if f_val < min_F:
                    min_F = f_val
                    min_loc = (lons[j], lats[i])
                    
        growth_rate = 0.0
        if t > 0:
            growth_rate = area - metrics["Area_km2"][-1]
            
        metrics["Year"].append(year)
        metrics["Center_Lon"].append(min_loc[0])
        metrics["Center_Lat"].append(min_loc[1])
        metrics["Min_F_nT"].append(min_F)
        metrics["Area_km2"].append(area)
        metrics["Growth_Rate_km2_yr"].append(growth_rate)
        
        if year % 10 == 0:
            print(f"Processed {year} | Area: {area / 1e6:,.2f} M km^2 | Min F: {min_F:.0f} nT")
            
    df = pd.DataFrame(metrics)
    
    os.makedirs("outputs/csv", exist_ok=True)
    df.to_csv("outputs/csv/saa_tracking.csv", index=False)
    print("Metrics saved to outputs/csv/saa_tracking.csv")
    
    # Plotting metrics
    os.makedirs("outputs/figures", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df['Year'], df['Area_km2'] / 1e6, 'b-', label='Area')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Area (Million km$^2$)', color='b')
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(df['Year'], df['Growth_Rate_km2_yr'] / 1e6, 'r--', label='Growth Rate', alpha=0.5)
    
    # Calculate a moving average for growth rate to smooth it out
    df['Growth_Rate_MA5'] = df['Growth_Rate_km2_yr'].rolling(window=5, center=True).mean()
    ax2.plot(df['Year'], df['Growth_Rate_MA5'] / 1e6, 'r-', linewidth=2, label='Growth Rate (5-yr MA)')
    
    ax2.set_ylabel('Growth Rate (Million km$^2$ / yr)', color='r')
    ax2.tick_params('y', colors='r')
    
    # Handle legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    plt.title("South Atlantic Anomaly Area & Growth Rate (F < 25,000 nT)")
    fig.tight_layout()
    plt.savefig("outputs/figures/saa_metrics.png", dpi=300)
    plt.close()
    
    # Create animation
    print("Generating map animation...")
    fig_map = plt.figure(figsize=(10, 6))
    fig_map.subplots_adjust(top=0.85)  # Make room for the multi-line title
    ax_map = fig_map.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    def update(frame):
        ax_map.clear()
        
        # Base map setup
        ax_map.add_feature(cfeature.LAND, facecolor='#E0E0E0')
        ax_map.add_feature(cfeature.OCEAN, facecolor='#FFFFFF')
        ax_map.coastlines(linewidth=0.5)
        
        # Add gridlines
        ax_map.gridlines(draw_labels=True, linestyle=':', color='gray', alpha=0.5)
        
        # Plot full contour map
        F_frame = all_F[frame]
        
        # Only plot contour if there are values below the threshold
        if np.nanmin(F_frame) < threshold:
            # Filled contour for the anomaly
            cf = ax_map.contourf(Lon, Lat, F_frame, levels=[0, threshold], 
                                 colors=['red'], alpha=0.3, transform=ccrs.PlateCarree())
                                 
            # Contour line for the anomaly
            cs = ax_map.contour(Lon, Lat, F_frame, levels=[threshold], 
                                colors=['darkred'], linewidths=2, transform=ccrs.PlateCarree())
                            
        # Plot center
        c_lon = metrics['Center_Lon'][frame]
        c_lat = metrics['Center_Lat'][frame]
        ax_map.plot(c_lon, c_lat, 'ko', markersize=6, transform=ccrs.PlateCarree())
        
        # Plot historical path up to this frame
        ax_map.plot(metrics['Center_Lon'][:frame+1], metrics['Center_Lat'][:frame+1], 
                    'k-', linewidth=1.5, alpha=0.7, transform=ccrs.PlateCarree())
                    
        year = years[frame]
        area = metrics['Area_km2'][frame] / 1e6
        ax_map.set_title(f"South Atlantic Anomaly Evolution ({year})\nArea: {area:.2f} Million km$^2$ | Center: {c_lat:.1f}°N, {c_lon:.1f}°E", pad=20)
        
        # Set extent to focus on the Atlantic/South America
        ax_map.set_extent([-120, 60, -90, 30], crs=ccrs.PlateCarree())

    ani = animation.FuncAnimation(fig_map, update, frames=len(years), interval=100, repeat_delay=2000)
    ani.save("outputs/figures/saa_animation.gif", writer='pillow', fps=10, dpi=150)
    print("Animation saved to outputs/figures/saa_animation.gif")

if __name__ == "__main__":
    main()
