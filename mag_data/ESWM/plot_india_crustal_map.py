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

shc_file = "SW_OPER_MLI_SHA_2C_00000000T000000_99999999T999999_1101/SW_OPER_MLI_SHA_2C_00000000T000000_99999999T999999_1101.shc"
time, coeffs, parameters = cp.data_utils.load_shcfile(shc_file)
coeffs = np.squeeze(coeffs)

lat_min, lat_max = 5, 40
lon_min, lon_max = 65, 100

theta = np.linspace(90 - lat_max, 90 - lat_min, 200) # 50 to 85. So 90-theta goes 40 to 5.
phi = np.linspace(lon_min, lon_max, 200)
phi_grid, theta_grid = np.meshgrid(phi, theta)
radius = 6371.2 * np.ones_like(theta_grid)

B_radius, _, _ = cp.model_utils.synth_values(coeffs, radius, theta_grid, phi_grid, nmax=parameters['nmax'], nmin=parameters['nmin'])
Z = -B_radius

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add borders and coastlines FIRST
ax.add_feature(cfeature.BORDERS, linewidth=1, linestyle=':')
ax.coastlines(linewidth=1.5)

# Plot Z with contourf
pcm = ax.contourf(phi_grid, 90 - theta_grid, Z, levels=50, cmap='RdBu_r', transform=ccrs.PlateCarree())

# Colorbar and titles
cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', shrink=0.8, pad=0.05)
cbar.set_label('Radial Magnetic Field Anomaly Z (nT)')
plt.title('Lithospheric Magnetic Field Over India\n(Swarm 2C Model, n=16 to 120)', pad=20)

plt.savefig('india_crustal_map_2C.png', dpi=300, bbox_inches='tight')
print("Map saved to india_crustal_map_2C.png")
