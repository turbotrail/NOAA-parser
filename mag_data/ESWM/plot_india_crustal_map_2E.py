# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chaosmagpy",
#     "matplotlib",
#     "numpy",
#     "cartopy"
# ]
# ///
import chaosmagpy as cp
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Use the 2E model which is ultra-high resolution (n=1500)
shc_file = "SW_OPER_MLI_SHA_2E_00000000T000000_99999999T999999_0901/SW_OPER_MLI_SHA_2E_00000000T000000_99999999T999999_0901.shc"

# Load coefficients
print("Loading 2E model...")
time, coeffs, parameters = cp.data_utils.load_shcfile(shc_file)

# The shape is (N_coeffs, 1) or similar. Make it 1D.
coeffs = np.squeeze(coeffs)

# India bounding box
lat_min, lat_max = 5, 40
lon_min, lon_max = 65, 100

# High-resolution grid for India
# Pass 1D arrays to synth_values with grid=True for highly optimized performance!
theta = np.linspace(90 - lat_max, 90 - lat_min, 400) # 50 to 85
phi = np.linspace(lon_min, lon_max, 400)             # 65 to 100

nmax = parameters['nmax']
nmin = parameters['nmin']

# Evaluate field using chaosmagpy with grid=True (ultra fast for large max_degree)
print(f"Synthesizing magnetic field up to degree {nmax} over India... please wait.")
B_radius, B_theta, B_phi = cp.model_utils.synth_values(coeffs, 6371.2, theta, phi, nmax=nmax, nmin=nmin, grid=True)

# Vertical component anomaly (Z = -B_radius)
Z = -B_radius

# Now we construct the 2D grids for matplotlib plotting
phi_grid, theta_grid = np.meshgrid(phi, theta)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add map features ON TOP
ax.add_feature(cfeature.BORDERS, linewidth=1, linestyle=':')
ax.coastlines(linewidth=1.5)

# Plot Z component using contourf which is extremely robust in Cartopy
print("Plotting the high resolution data...")
pcm = ax.contourf(phi_grid, 90 - theta_grid, Z, levels=100, 
                  transform=ccrs.PlateCarree(), cmap='RdBu_r', extend='both')

# Add gridlines
gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl.top_labels = False
gl.right_labels = False

cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', shrink=0.8, pad=0.05)
cbar.set_label('Radial Magnetic Field Anomaly Z (nT)')
plt.title('Lithospheric Magnetic Field Over India\n(Swarm 2E High-Res Model, n=16 to 1500)', pad=20)

plt.savefig('india_crustal_map_2E.png', dpi=400, bbox_inches='tight')
print("Map saved to india_crustal_map_2E.png")
