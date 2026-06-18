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

shc_file = "SW_OPER_MLI_SHA_2C_00000000T000000_99999999T999999_1101/SW_OPER_MLI_SHA_2C_00000000T000000_99999999T999999_1101.shc"

# Load coefficients
time, coeffs, parameters = cp.data_utils.load_shcfile(shc_file)

print(f"coeffs shape: {coeffs.shape}")
print(f"parameters: {parameters}")

# coeffs shape is (14385, 1), we need a 1D array (14385,)
coeffs = np.squeeze(coeffs)

print(f"coeffs shape after modification: {coeffs.shape}")

# Setup grid
theta = np.linspace(1, 179, 180)
phi = np.linspace(-180, 180, 360)
phi_grid, theta_grid = np.meshgrid(phi, theta)
radius = 6371.2 * np.ones_like(theta_grid)

nmax = parameters['nmax']
nmin = parameters['nmin']

# Evaluate field using chaosmagpy
# Base model evaluator:
# synth_values(coeffs, radius, theta, phi, nmax=None, nmin=None, source='internal')
B_radius, B_theta, B_phi = cp.model_utils.synth_values(coeffs, radius, theta_grid, phi_grid, nmax=nmax, nmin=nmin)

# Vertical component anomaly (Z = -B_radius)
Z = -B_radius

fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.Robinson())
ax.coastlines(linewidth=0.5, color='black', alpha=0.5)
ax.gridlines(linestyle='--', alpha=0.5)

# Plot Z component
pcm = ax.pcolormesh(phi_grid, 90 - theta_grid, Z, transform=ccrs.PlateCarree(),
                    cmap='RdBu_r', vmin=-100, vmax=100)
cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
cbar.set_label('Radial Magnetic Field Anomaly Z (nT)')
plt.title('Lithospheric Magnetic Field (Swarm 2C Model, n=16 to 120)')

plt.savefig('lithospheric_map_2C.png', dpi=300, bbox_inches='tight')
print("Map saved to lithospheric_map_2C.png")
