import numpy as np
import os
import chaosmagpy as cp
import time

shc_file = r"c:\Users\Jeyaprakash S\Documents\GitHub\NOAA-parser\mag_data\ESWM\SW_OPER_MLI_SHA_2E_00000000T000000_99999999T999999_0901\SW_OPER_MLI_SHA_2E_00000000T000000_99999999T999999_0901.shc"

def fast_load_shcfile(filepath, leap_year=None, comment=None):
    from chaosmagpy.data_utils import dyear_to_mjd
    leap_year = True if leap_year is None else leap_year
    comment = '#' if comment is None else comment

    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith(comment):
                continue
            header_line = line
            break
        
        for line in f:
            if line.strip().startswith(comment):
                continue
            time_line = line
            break
        
        newline = np.fromstring(header_line, sep=' ')
        name = os.path.split(filepath)[1]
        values = [name] + newline.astype(int).tolist()
        
        keys = ['SHC', 'nmin', 'nmax', 'N', 'order', 'step']
        parameters = dict(zip(keys, values))
        
        times = np.fromstring(time_line, sep=' ')
        coeffs_data = np.loadtxt(f, comments=comment)
        coeffs = coeffs_data[:, 2:]
        mjd = dyear_to_mjd(times, leap_year=leap_year)
        
    return mjd, coeffs, parameters

def fast_synth_values(coeffs, radius, theta, phi, nmax=None, nmin=None, mmax=None, source=None):
    # Specialized fast synth_values for internal field, grid=True, single time step
    from chaosmagpy.model_utils import legendre_poly
    
    nmin = 1 if nmin is None else int(nmin)
    dim = coeffs.shape[-1]
    
    if nmax is None:
        nmax = int(np.sqrt(dim + nmin**2) - 1)
    mmax = nmax if mmax is None else mmax
    
    # We assume source='internal' and radius is a scalar, e.g. 6371.2
    # Normalize radius
    r_surf = 6371.2
    radius_norm = radius / r_surf
    
    # 1. Populating g and h matrices: shape (nmax+1, nmax+1)
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
            
    # 2. Compute Legendre polynomials
    # Pnm shape: (nmax + 1, nmax + 2, len(theta))
    # Note: we pass 1D theta to legendre_poly to get a 3D array (nmax+1, nmax+2, len(theta))
    Pnm = legendre_poly(nmax, theta)
    sinth = Pnm[1, 1]
    
    # 3. Precompute sin/cos for phi
    phi_rad = np.radians(phi)
    cos_m_phi = [None] + [np.cos(m * phi_rad) for m in range(1, mmax + 1)]
    sin_m_phi = [None] + [np.sin(m * phi_rad) for m in range(1, mmax + 1)]
    
    # 4. Allocate outputs
    B_radius = np.zeros((len(theta), len(phi)))
    B_theta = np.zeros((len(theta), len(phi)))
    B_phi = np.zeros((len(theta), len(phi)))
    
    # 5. Precompute r_n values for all n
    r_n_all = radius_norm ** (-np.arange(nmax + 1) - 2)
    
    # 6. Summation over m (0 to mmax)
    for m in range(mmax + 1):
        n_vals = np.arange(max(nmin, m), nmax + 1)
        if len(n_vals) == 0:
            continue
            
        r_n_vals = r_n_all[n_vals]
        
        # B_radius factor: (n + 1) * r_n_vals
        factor_r = (n_vals + 1) * r_n_vals
        # B_theta factor: -r_n_vals
        factor_theta = -r_n_vals
        # B_phi factor: r_n_vals
        factor_phi = r_n_vals
        
        g_coeffs = g[n_vals, m]
        
        # Pnm_m shape: (len(n_vals), len(theta))
        Pnm_m = Pnm[n_vals, m, :]
        
        # dPnm_m shape: (len(n_vals), len(theta))
        # dP_n^m/d_theta is at Pnm[m, n+1]
        dPnm_m = Pnm[m, n_vals + 1, :]
        
        # Sum over n
        # U_m = sum_n factor * g_coeffs * Pnm_m
        U_m_r = np.sum((factor_r * g_coeffs)[:, None] * Pnm_m, axis=0)
        U_m_theta = np.sum((factor_theta * g_coeffs)[:, None] * dPnm_m, axis=0)
        
        if m == 0:
            B_radius += np.outer(U_m_r, np.ones(len(phi)))
            B_theta += np.outer(U_m_theta, np.ones(len(phi)))
        else:
            h_coeffs = h[n_vals, m]
            
            V_m_r = np.sum((factor_r * h_coeffs)[:, None] * Pnm_m, axis=0)
            V_m_theta = np.sum((factor_theta * h_coeffs)[:, None] * dPnm_m, axis=0)
            
            cos_val = cos_m_phi[m]
            sin_val = sin_m_phi[m]
            
            B_radius += np.outer(U_m_r, cos_val) + np.outer(V_m_r, sin_val)
            B_theta += np.outer(U_m_theta, cos_val) + np.outer(V_m_theta, sin_val)
            
            # B_phi component
            div_Pnm = Pnm_m / sinth[None, :]
            U_m_phi = np.sum((factor_phi * g_coeffs)[:, None] * div_Pnm, axis=0)
            V_m_phi = np.sum((factor_phi * h_coeffs)[:, None] * div_Pnm, axis=0)
            
            B_phi += m * (np.outer(U_m_phi, sin_val) - np.outer(V_m_phi, cos_val))
            
    return B_radius, B_theta, B_phi

# Let's benchmark it
print("Loading 2E model...")
mjd, coeffs, parameters = fast_load_shcfile(shc_file)
coeffs = np.squeeze(coeffs)

# User's grid: 400x400
theta = np.linspace(90 - 40, 90 - 5, 400)
phi = np.linspace(65, 100, 400)

nmax = parameters['nmax']
nmin = parameters['nmin']

print(f"Synthesizing magnetic field on 400x400 grid using FAST custom synthesis...")
t0 = time.time()
B_radius, B_theta, B_phi = fast_synth_values(coeffs, 6371.2, theta, phi, nmax=nmax, nmin=nmin)
t1 = time.time()
print(f"Fast Custom Synthesis took: {t1 - t0:.4f} seconds")

# Verify correctness against chaosmagpy for a small 10x10 subset
print("Verifying correctness on a 10x10 grid...")
theta_sub = theta[::40]
phi_sub = phi[::40]

B_r_ref, B_t_ref, B_p_ref = cp.model_utils.synth_values(coeffs, 6371.2, theta_sub, phi_sub, nmax=nmax, nmin=nmin, grid=True)
B_r_fast, B_t_fast, B_p_fast = fast_synth_values(coeffs, 6371.2, theta_sub, phi_sub, nmax=nmax, nmin=nmin)

assert np.allclose(B_r_ref, B_r_fast), "B_radius mismatch!"
assert np.allclose(B_t_ref, B_t_fast), "B_theta mismatch!"
assert np.allclose(B_p_ref, B_p_fast), "B_phi mismatch!"
print("All values matched perfectly!")
