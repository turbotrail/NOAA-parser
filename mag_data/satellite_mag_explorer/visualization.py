import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import matplotlib.animation as animation

def plot_ground_track_with_bfield(mag_df: pd.DataFrame, title: str = "Satellite Ground Track"):
    """
    Plots the satellite ground track colored by total magnetic field strength (F).
    """
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, color='gray', alpha=0.5)

    scatter = ax.scatter(mag_df['Longitude'], mag_df['Latitude'], c=mag_df['F'], 
                         cmap='viridis', s=5, transform=ccrs.PlateCarree())
    
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label('Total Magnetic Field Strength (F) [nT]')
    ax.set_title(title)
    return fig

def plot_time_series(mag_df: pd.DataFrame):
    """
    Plots time series of magnetic field, inclination, declination, and altitude.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    axs[0].plot(mag_df['Time'], mag_df['F'], label='Total Field (F)', color='blue')
    axs[0].set_ylabel('F [nT]')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)
    
    axs[1].plot(mag_df['Time'], mag_df['Inc'], label='Inclination', color='orange')
    axs[1].plot(mag_df['Time'], mag_df['Dec'], label='Declination', color='green')
    axs[1].set_ylabel('Degrees')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)
    
    axs[2].plot(mag_df['Time'], mag_df['Altitude_km'], label='Altitude', color='red')
    axs[2].set_ylabel('Altitude [km]')
    axs[2].set_xlabel('Time')
    axs[2].legend(loc='upper right')
    axs[2].grid(True)
    
    plt.tight_layout()
    return fig

def animate_orbit(mag_df: pd.DataFrame, output_path: str, fps: int = 15):
    """
    Generates an MP4 animation of the satellite orbit colored by F-field.
    """
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.gridlines(draw_labels=True, color='gray', alpha=0.5)
    
    # Plot entire track lightly
    ax.plot(mag_df['Longitude'], mag_df['Latitude'], color='lightgray', transform=ccrs.PlateCarree(), zorder=1)
    
    scatter = ax.scatter([], [], c=[], cmap='viridis', s=30, vmin=mag_df['F'].min(), vmax=mag_df['F'].max(), transform=ccrs.PlateCarree(), zorder=2)
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label('Total Magnetic Field Strength (F) [nT]')
    
    title = ax.set_title('')
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        return scatter, title
        
    def update(frame):
        # Show trail of last 20 points
        start_idx = max(0, frame - 20)
        subset = mag_df.iloc[start_idx:frame+1]
        
        # update scatter
        offsets = np.column_stack((subset['Longitude'], subset['Latitude']))
        scatter.set_offsets(offsets)
        scatter.set_array(subset['F'])
        
        time_str = mag_df.iloc[frame]['Time'].strftime('%Y-%m-%d %H:%M:%S')
        title.set_text(f"Time: {time_str}")
        return scatter, title

    ani = animation.FuncAnimation(fig, update, frames=len(mag_df), init_func=init, blit=False)
    ani.save(output_path, writer='ffmpeg', fps=fps)
    plt.close(fig)
