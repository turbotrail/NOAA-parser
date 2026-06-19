# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib", "cartopy", "chaosmagpy"]
# ///
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import chaosmagpy as cp
import sys

def parse_cfw_file(filepath):
    times = []
    t_coeffs = {}
    p_coeffs = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) == 3 and not parts[0].isalpha():
            min_sh = int(parts[0])
            max_sh = int(parts[1])
            n_times = int(parts[2])
        elif len(parts) == n_times:
            times = [float(x) for x in parts]
        elif parts[0] == 'T':
            l, m = int(parts[1]), int(parts[2])
            coeffs = np.array([float(x) for x in parts[3:]])
            t_coeffs[(l, m)] = coeffs
        elif parts[0] == 'P':
            l, m = int(parts[1]), int(parts[2])
            coeffs = np.array([float(x) for x in parts[3:]])
            p_coeffs[(l, m)] = coeffs
            
    return times, max_sh, t_coeffs, p_coeffs

def dict_to_array(coeffs_dict, max_sh, time_idx):
    # chaosmagpy expects a 1D array of coefficients ordered like:
    # g_1^0, g_1^1, h_1^1, g_2^0, g_2^1, h_2^1, g_2^2, h_2^2, ...
    n_coeffs = max_sh * (max_sh + 2)
    coeffs_array = np.zeros(n_coeffs)
    for l in range(1, max_sh + 1):
        for m in range(-l, l + 1):
            if (l, m) in coeffs_dict:
                val = coeffs_dict[(l, m)][time_idx]
                if m == 0:
                    idx = l**2 - 1
                elif m > 0:
                    idx = l**2 - 1 + 2*m - 1
                else: # m < 0
                    idx = l**2 - 1 + 2*abs(m)
                coeffs_array[idx] = val
    return coeffs_array

def main():
    filepath = 'SW_OPER_CFW_SHA_2Y_19990101T000000_20240731T000000_1101.txt'
    print(f"Parsing {filepath}...")
    times, max_sh, t_coeffs, p_coeffs = parse_cfw_file(filepath)
    
    # Use the latest time index
    time_idx = -1
    year = times[time_idx]
    print(f"Plotting for year {year}")
    
    t_arr = dict_to_array(t_coeffs, max_sh, time_idx)
    p_arr = dict_to_array(p_coeffs, max_sh, time_idx)
    
def plot_region(t_arr, p_arr, year, lon_bounds, lat_bounds, filename, title_suffix):
    lon_min, lon_max = lon_bounds
    lat_min, lat_max = lat_bounds
    
    # Create a grid for the region
    lon = np.linspace(lon_min, lon_max, 100)
    lat = np.linspace(lat_min, lat_max, 100)
    theta = 90.0 - lat
    phi = lon
    
    grid_theta, grid_phi = np.meshgrid(theta, phi, indexing='ij')
    radius = 6371.2 * np.ones_like(grid_theta)
    
    # Evaluate potentials
    _, B_theta_P, B_phi_P = cp.model_utils.synth_values(p_arr, radius, grid_theta, grid_phi)
    _, B_theta_T, B_phi_T = cp.model_utils.synth_values(t_arr, radius, grid_theta, grid_phi)
    
    u_theta = -B_theta_P - B_phi_T
    u_phi = -B_phi_P + B_theta_T
    
    v_north = -u_theta
    v_east = u_phi
    speed = np.sqrt(v_north**2 + v_east**2)
    
    plt.figure(figsize=(10, 8))
    
    # Use an appropriate projection for the region
    center_lon = (lon_min + lon_max) / 2
    center_lat = (lat_min + lat_max) / 2
    ax = plt.axes(projection=ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat))
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.coastlines(color='gray', alpha=0.8, resolution='50m')
    
    grid_lon, grid_lat = np.meshgrid(lon, lat)
    
    pcm = ax.contourf(grid_lon, grid_lat, speed, levels=20, transform=ccrs.PlateCarree(), cmap='viridis')
    plt.colorbar(pcm, label='Core Flow Speed (km/yr)', shrink=0.7)
    
    skip = 4
    ax.quiver(grid_lon[::skip, ::skip], grid_lat[::skip, ::skip], 
              v_east[::skip, ::skip], v_north[::skip, ::skip], 
              transform=ccrs.PlateCarree(), color='white', 
              width=0.003, headwidth=4, headlength=5)
              
    plt.title(f'Swarm Core Flow Model (Epoch {year:.2f}) - {title_suffix}')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def main():
    filepath = 'SW_OPER_CFW_SHA_2Y_19990101T000000_20240731T000000_1101.txt'
    print(f"Parsing {filepath}...")
    times, max_sh, t_coeffs, p_coeffs = parse_cfw_file(filepath)
    
    time_idx = -1
    year = times[time_idx]
    print(f"Plotting for year {year}")
    
    t_arr = dict_to_array(t_coeffs, max_sh, time_idx)
    p_arr = dict_to_array(p_coeffs, max_sh, time_idx)
    
    # Plot India
    plot_region(t_arr, p_arr, year, lon_bounds=[65, 100], lat_bounds=[5, 40], filename='swarm_cfw_india.png', title_suffix='India')
    
    # Plot SAA
    plot_region(t_arr, p_arr, year, lon_bounds=[-90, 40], lat_bounds=[-50, 0], filename='swarm_cfw_saa.png', title_suffix='South Atlantic Anomaly')
    
    # Plot Indian Ocean
    plot_region(t_arr, p_arr, year, lon_bounds=[30, 115], lat_bounds=[-50, 25], filename='swarm_cfw_indian_ocean.png', title_suffix='Indian Ocean')

if __name__ == '__main__':
    main()
