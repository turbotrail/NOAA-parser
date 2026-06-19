# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pyshtools",
#     "matplotlib",
#     "numpy",
#     "cartopy"
# ]
# ///
import pyshtools as pysh
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Plot EGF Gravity Anomaly map over India")
parser.add_argument('--nmin', type=int, default=0, help='Minimum degree (default 0)')
parser.add_argument('--nmax', type=int, default=40, help='Maximum degree (default 40)')
args = parser.parse_args()

gfc_file = "SW_OPER_EGF_SHA_2__20251201T000000_20251231T235959_0101/SW_OPER_EGF_SHA_2__20251201T000000_20251231T235959_0101.gfc"
print("Loading EGF gravity model...")
clm = pysh.SHGravCoeffs.from_file(gfc_file, format='icgem')

# Cap nmax at the model's actual maximum degree (40 for this EGF model)
nmin = args.nmin
nmax = min(args.nmax, clm.lmax)

# Create a filtered copy of the coefficients
coeffs_filtered = clm.copy()

# Zero out degrees below nmin
if nmin > 0:
    for l in range(0, nmin):
        if l <= clm.lmax:
            coeffs_filtered.coeffs[:, l, :] = 0.0

# Zero out degrees above nmax
if nmax < clm.lmax:
    for l in range(nmax + 1, clm.lmax + 1):
        if l <= clm.lmax:
            coeffs_filtered.coeffs[:, l, :] = 0.0

print(f"Synthesizing gravity disturbance from degree {nmin} up to degree {nmax} over India... please wait.")

# Expand gravity globally (spherical approximation, normal gravity removed by default in pyshtools)
# .expand() returns an SHGravGrid object
grids = coeffs_filtered.expand(a=clm.r0, f=0, lmax=clm.lmax)

# Extract radial gravity disturbance (Z component) in mGal
# Note: pyshtools returns gravity components in SI units (m/s^2). 1 m/s^2 = 10^5 mGal.
# The `rad` attribute is the radial component (positive upwards). 
# We'll invert it to make it positive downwards like typical gravity anomalies.
global_z_mgal = -grids.rad.to_array() * 1e5
lats = grids.rad.lats()
lons = grids.rad.lons()

# Filter for India's bounding box: lat 5 to 40, lon 65 to 100
lat_mask = (lats >= 5) & (lats <= 40)
lon_mask = (lons >= 65) & (lons <= 100)

Z_india = global_z_mgal[lat_mask, :][:, lon_mask]
lats_india = lats[lat_mask]
lons_india = lons[lon_mask]

phi_grid, theta_grid = np.meshgrid(lons_india, lats_india)

# --- Plotting ---
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

print("Plotting the gravity data...")
vmax = np.max(np.abs(Z_india))
if vmax == 0: vmax = 1.0
levels = np.linspace(-vmax * 1.05, vmax * 1.05, 100)

pcm = ax.contourf(phi_grid, theta_grid, Z_india, levels=levels, 
                  transform=ccrs.PlateCarree(), cmap='RdBu_r', extend='both')

ax.add_feature(cfeature.BORDERS, linewidth=1, linestyle=':')
ax.coastlines(linewidth=1.5)

gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl.top_labels = False
gl.right_labels = False

cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', shrink=0.8, pad=0.05)
cbar.set_label(f'Radial Gravity Disturbance (mGal) {nmin}_{nmax}')

plt.title(f'Gravity Field Over India\n(Swarm EGF Model, n={nmin} to {nmax})\nAmplitude Range: {np.min(Z_india):.1f} mGal to {np.max(Z_india):.1f} mGal', pad=20)

output_file = f'india_gravity_map_EGF_{nmin}_{nmax}.png'
plt.savefig(output_file, dpi=300)
print(f"Map saved to {output_file}")
