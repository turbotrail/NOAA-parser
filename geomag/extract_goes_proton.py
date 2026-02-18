import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
DATA_DIR = Path("goes19-proton")
SENSOR = 0  # usually 0 or 1; we can average later

# =========================
# LOAD ALL FILES
# =========================
files = sorted(DATA_DIR.glob("*.nc"))
ds = xr.open_mfdataset(
    files,
    combine="nested",
    concat_dim="report_number",
    parallel=True
)


# =========================
# EXTRACT TIME
# =========================
time = ds.L1a_SciData_TimeStamp.sel(sensor_unit=SENSOR).values

# =========================
# EXTRACT DIFFERENTIAL PROTON FLUX (T3 – HIGH ENERGY)
# =========================
flux_T3 = ds.T3_DifferentialProtonFluxes.sel(sensor_unit=SENSOR)
dqf_T3  = ds.T3_DifferentialProtonFluxDQFs.sel(sensor_unit=SENSOR)

print("T3 Differential Flux stats:")
print("min:", float(flux_T3.min()))
print("max:", float(flux_T3.max()))

# Accept nominal + corrected data
#
# Relax DQF filtering: keep all science data except explicitly bad
flux_T3 = flux_T3.where(dqf_T3 < 4)

# Replace zeros with NaN only for plotting safety
flux_T3 = flux_T3.where(flux_T3 != 0)

# Clip for log-plot visibility (visualization only)
flux_T3_plot = flux_T3.clip(min=1e-7)

# =========================
# TIME SERIES 1: MEAN T3 FLUX
# =========================
flux_T3_mean = flux_T3_plot.mean(dim="energy_T3")

# Force computation to avoid Dask reduction issues
flux_T3_mean = flux_T3_mean.compute()

# Absolute fallback: if everything is NaN, use raw mean (instrument background)
if np.all(np.isnan(flux_T3_mean.values)):
    print("⚠️ All-NaN after DQF filtering — falling back to unfiltered data")
    flux_T3_mean = ds.T3_DifferentialProtonFluxes.sel(sensor_unit=SENSOR).mean(dim="energy_T3").compute()

# =========================
# PLOT 1: LINEAR SCALE (QUIET-TIME VISIBILITY)
# =========================
plt.figure(figsize=(14, 5))
plt.plot(time, flux_T3_mean, lw=1, marker=".", markersize=1)

plt.yscale("linear")
plt.ylim(0, 3e-4)

plt.xlabel("Time (UTC)")
plt.ylabel("Mean T3 Differential Proton Flux")
plt.title("GOES-19 SGPS – Mean High-Energy Proton Flux (Linear Scale)")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# PLOT 2: BACKGROUND-SUBTRACTED ANOMALY
# =========================
valid = flux_T3_mean.values[np.isfinite(flux_T3_mean.values)]
background = float(np.median(valid)) if valid.size > 0 else 0.0
flux_anomaly = flux_T3_mean - background

plt.figure(figsize=(14, 5))
plt.plot(time, flux_anomaly, lw=1)

plt.axhline(0, color="k", ls="--", alpha=0.5)
plt.xlabel("Time (UTC)")
plt.ylabel("Δ Proton Flux")
plt.title("GOES-19 SGPS – T3 Proton Flux Anomaly")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# PLOT 3: SMOOTHED TIME SERIES
# =========================
flux_smooth = flux_T3_mean.rolling(report_number=60, center=True).mean()

plt.figure(figsize=(14, 5))
plt.plot(time, flux_smooth, lw=2)

plt.yscale("linear")
plt.ylim(0, 3e-4)

plt.xlabel("Time (UTC)")
plt.ylabel("Smoothed Mean T3 Flux")
plt.title("GOES-19 SGPS – Smoothed High-Energy Proton Flux")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# TIME SERIES 4: HIGHEST-ENERGY T3 CHANNEL (QUIET-SAFE)
# =========================
flux_T3_high = flux_T3.isel(energy_T3=-1).compute()

# Fallback if channel is all NaN or zero
if np.all(~np.isfinite(flux_T3_high.values)):
    print("⚠️ Highest-energy T3 channel is empty (quiet background)")
    flux_T3_high = ds.T3_DifferentialProtonFluxes.sel(
        sensor_unit=SENSOR
    ).isel(energy_T3=-1).compute()

plt.figure(figsize=(14, 5))
plt.plot(time, flux_T3_high, lw=1, marker=".", markersize=1)

plt.yscale("linear")
plt.ylim(0, 3e-4)

plt.xlabel("Time (UTC)")
plt.ylabel("Differential Proton Flux")
plt.title("GOES-19 SGPS – Highest-Energy Proton Channel (Linear Scale)")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()