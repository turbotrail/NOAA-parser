import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
DATA_DIR = Path("goes19-mps")   # folder with .nc files

# =========================
# LOAD FILES
# =========================
files = sorted(DATA_DIR.glob("*.nc"))

ds = xr.open_mfdataset(
    files,
    combine="nested",
    concat_dim="report_number"
)

# =========================
# TIME
# =========================
time = ds.L1a_SciData_TimeStamp.values

# =========================
# ELECTRON FLUX
# =========================
flux_e = ds.DiffElectronFluxes
dqf_e  = ds.DiffElectronFluxDQFs

print("Electron flux stats:")
print("min:", float(flux_e.min()))
print("max:", float(flux_e.max()))

# Keep calibrated values (can be negative in quiet times)
flux_e = flux_e.where(dqf_e < 4)

# =========================
# MEAN ELECTRON FLUX
# (average over energy + FOV)
# =========================
flux_e_mean = (
    flux_e
    .mean(dim="differential_flux_energy_band")
    .mean(dim="field_of_view")
    .compute()
)

# Background-subtracted anomaly (reveals negative excursions)
valid = flux_e_mean.values[np.isfinite(flux_e_mean.values)]
background = np.median(valid) if valid.size > 0 else 0.0
flux_e_anomaly = flux_e_mean - background

# =========================
# PLOT 1: MEAN ELECTRONS
# =========================
plt.figure(figsize=(14, 5))
plt.plot(time, flux_e_mean, lw=1, marker=".", markersize=1)

plt.yscale("linear")
plt.xlabel("Time (UTC)")
plt.ylabel("Mean Electron Flux")
plt.title("GOES-19 MPS-LO – Mean Magnetospheric Electron Flux")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# PLOT 1b: ELECTRON FLUX ANOMALY (NEGATIVE VISIBLE)
# =========================
plt.figure(figsize=(14, 5))
plt.plot(time, flux_e_anomaly, lw=1, marker=".", markersize=1)

max_abs = np.nanmax(np.abs(flux_e_anomaly.values))
plt.ylim(-max_abs, max_abs)

plt.axhline(0, color="k", ls="--", alpha=0.5)
plt.xlabel("Time (UTC)")
plt.ylabel("Δ Electron Flux")
plt.title("GOES-19 MPS-LO – Electron Flux Anomaly (Background Subtracted)")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# HIGHEST-ENERGY ELECTRON BAND
# =========================
flux_e_high = (
    flux_e
    .isel(differential_flux_energy_band=-1)
    .mean(dim="field_of_view")
    .compute()
)

# Fallback if quiet
if np.all(~np.isfinite(flux_e_high.values)):
    print("⚠️ High-energy electron channel is quiet")

# =========================
# PLOT 2: HIGH-ENERGY ELECTRONS
# =========================
plt.figure(figsize=(14, 5))
plt.plot(time, flux_e_high, lw=1, marker=".", markersize=1)

plt.yscale("linear")
plt.xlabel("Time (UTC)")
plt.ylabel("High-Energy Electron Flux")
plt.title("GOES-19 MPS-LO – Highest-Energy Electron Band")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()