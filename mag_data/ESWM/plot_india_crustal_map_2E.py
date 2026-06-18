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

import os

def fast_load_shcfile(filepath, leap_year=None, comment=None):
    from chaosmagpy.data_utils import dyear_to_mjd
    leap_year = True if leap_year is None else leap_year
    comment = '#' if comment is None else comment

    with open(filepath, 'r') as f:
        # read header line
        for line in f:
            if line.strip().startswith(comment):
                continue
            header_line = line
            break
        
        # read times line
        for line in f:
            if line.strip().startswith(comment):
                continue
            time_line = line
            break
        
        # unpack parameter line
        newline = np.fromstring(header_line, sep=' ')
        name = os.path.split(filepath)[1]
        values = [name] + newline.astype(int).tolist()
        
        keys = ['SHC', 'nmin', 'nmax', 'N', 'order', 'step']
        parameters = dict(zip(keys, values))
        
        # read times
        times = np.fromstring(time_line, sep=' ')
        
        # read coefficients using numpy loadtxt (fast block read)
        coeffs_data = np.loadtxt(f, comments=comment)
        
        coeffs = coeffs_data[:, 2:]
        mjd = dyear_to_mjd(times, leap_year=leap_year)
        
    return mjd, coeffs, parameters

def fast_synth_radius_only(coeffs, radius, theta, phi, nmax=None, nmin=None, mmax=None):
    from chaosmagpy.model_utils import legendre_poly
    
    nmin = 1 if nmin is None else int(nmin)
    dim = coeffs.shape[-1]
    
    if nmax is None:
        nmax = int(np.sqrt(dim + nmin**2) - 1)
    mmax = nmax if mmax is None else mmax
    
    r_surf = 6371.2
    radius_norm = radius / r_surf
    
    # Preallocate g and h arrays
    g = np.zeros((nmax + 1, nmax + 1))
    h = np.zeros((nmax + 1, nmax + 1))
    
    num = 0
    for n in range(nmin, nmax + 1):
        g[n, 0] = coeffs[num]
        num += 1
        for m in range(1, n + 1):
            g[n, m] = coeffs[num]
            h[n, m] = coeffs[num + 1]
            num += 2
            
    # Compute Legendre polynomials
    Pnm = legendre_poly(nmax, theta)
    
    phi_rad = np.radians(phi)
    cos_m_phi = [None] + [np.cos(m * phi_rad) for m in range(1, mmax + 1)]
    sin_m_phi = [None] + [np.sin(m * phi_rad) for m in range(1, mmax + 1)]
    
    # Accumulator for B_radius
    B_radius = np.zeros((len(theta), len(phi)))
    r_n_all = radius_norm ** (-np.arange(nmax + 1) - 2)
    
    for m in range(mmax + 1):
        n_vals = np.arange(max(nmin, m), nmax + 1)
        if len(n_vals) == 0:
            continue
            
        r_n_vals = r_n_all[n_vals]
        factor_r = (n_vals + 1) * r_n_vals
        g_coeffs = g[n_vals, m]
        Pnm_m = Pnm[n_vals, m, :]
        
        U_m_r = np.sum((factor_r * g_coeffs)[:, None] * Pnm_m, axis=0)
        
        if m == 0:
            B_radius += np.outer(U_m_r, np.ones(len(phi)))
        else:
            h_coeffs = h[n_vals, m]
            V_m_r = np.sum((factor_r * h_coeffs)[:, None] * Pnm_m, axis=0)
            
            B_radius += np.outer(U_m_r, cos_m_phi[m]) + np.outer(V_m_r, sin_m_phi[m])
            
    return B_radius

# Use the 2E model which is ultra-high resolution (n=1500)
shc_file = "SW_OPER_MLI_SHA_2E_00000000T000000_99999999T999999_0901/SW_OPER_MLI_SHA_2E_00000000T000000_99999999T999999_0901.shc"

# Load coefficients
print("Loading 2E model...")
time, coeffs, parameters = fast_load_shcfile(shc_file)

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
B_radius = fast_synth_radius_only(coeffs, 6371.2, theta, phi, nmax=nmax, nmin=nmin)

# Vertical component anomaly (Z = -B_radius)
Z = -B_radius

# Now we construct the 2D grids for matplotlib plotting
phi_grid, theta_grid = np.meshgrid(phi, theta)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
# Plot Z component using contourf first (so features are drawn on top)
print("Plotting the high resolution data...")
levels = np.linspace(-300, 300, 100)
pcm = ax.contourf(phi_grid, 90 - theta_grid, Z, levels=levels, 
                  transform=ccrs.PlateCarree(), cmap='RdBu_r', extend='both')

# Add map features ON TOP of the contour plot
ax.add_feature(cfeature.BORDERS, linewidth=1, linestyle=':')
ax.coastlines(linewidth=1.5)

# Add gridlines
gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl.top_labels = False
gl.right_labels = False

cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', shrink=0.8, pad=0.05)
cbar.set_label('Radial Magnetic Field Anomaly Z (nT)')
plt.title('Lithospheric Magnetic Field Over India\n(Swarm 2E High-Res Model, n=16 to 1500)', pad=20)

# Save the figure (removed bbox_inches='tight' to avoid map collapse)
plt.savefig('india_crustal_map_2E.png', dpi=300)
print("Map saved to india_crustal_map_2E.png")
