import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
DATA_DIR = Path("aditya-l1")   # folder with .nc files
USE_SENSOR = "2"               # MAG1 (science sensor)
RESAMPLE = "1min"              # None or e.g. "1min"

# =========================
# LOAD & STITCH
# =========================
dfs = []

for nc_file in sorted(DATA_DIR.glob("*.nc")):
    print(f"Reading {nc_file.name}")

    ds = xr.open_dataset(
        nc_file,
        decode_cf=True,
        mask_and_scale=True
    )

    # df = pd.DataFrame({
    #     "time": pd.to_datetime(ds["time"].to_numpy(), unit="s", utc=True),
    #     "Bx": ds[f"Bx{USE_SENSOR}_gsm"].values,
    #     "By": ds[f"By{USE_SENSOR}_gsm"].values,
    #     "Bz": ds[f"Bz{USE_SENSOR}_gsm"].values,
    # })
    df = pd.DataFrame({
        "time": pd.to_datetime(ds["time"].to_numpy(), unit="s", utc=True),
        "Bx": ds[f"B_yaw_mag{USE_SENSOR}"].values,
        "By": ds[f"B_roll_mag{USE_SENSOR}"].values,
        "Bz": ds[f"B_pitch_mag{USE_SENSOR}"].values,
    })

    dfs.append(df)
    ds.close()

# Concatenate all files
df_all = pd.concat(dfs, ignore_index=True)

# =========================
# CLEAN & SORT
# =========================
df_all.dropna(inplace=True)
df_all.sort_values("time", inplace=True)
df_all.drop_duplicates(subset="time", inplace=True)
df_all.set_index("time", inplace=True)

print(f"\nTotal samples: {len(df_all):,}")
print(f"Time span: {df_all.index.min()} → {df_all.index.max()}")

print(df_all.describe())
print("Unique Bx values:", pd.unique(df_all["Bx"])[:5])

# =========================
# OPTIONAL DOWNSAMPLING
# =========================
if RESAMPLE:
    df_all = df_all.resample(RESAMPLE).mean()
    print(f"Resampled to {RESAMPLE}, samples: {len(df_all):,}")

print(df_all.index.to_series().diff().value_counts().head())
# =========================
# PLOT
# =========================

print(df_all.isna().sum())

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

axes[0].plot(df_all.index, df_all["Bx"], linewidth=0.6)
axes[0].set_ylabel("Bx yaw) [nT]")
axes[0].grid(alpha=0.3)

axes[1].plot(df_all.index, df_all["By"], linewidth=0.6)
axes[1].set_ylabel("By roll [nT]")
axes[1].grid(alpha=0.3)

axes[2].plot(df_all.index, df_all["Bz"], linewidth=0.6)
axes[2].set_ylabel("Bz pitch [nT]")
axes[2].set_xlabel("Time (UTC)")
axes[2].grid(alpha=0.3)

plt.suptitle("Aditya-L1 MAG1 Components (Spacecraft frame)")
plt.tight_layout()
plt.show()


import numpy as np

df_all["Bmag"] = np.sqrt(
    df_all["Bx"]**2 +
    df_all["By"]**2 +
    df_all["Bz"]**2
)

# d|B|/dt (nT/s) — using 1-minute cadence
df_all["dBdt"] = df_all["Bmag"].diff() / 60.0

plt.figure(figsize=(16,4))
plt.plot(df_all.index, df_all["Bmag"], linewidth=0.6)
plt.ylabel("|B| (nT)")
plt.title("Aditya-L1 MAG (Spacecraft frame)")
plt.grid(alpha=0.3)
plt.show()

# Plot d|B|/dt
plt.figure(figsize=(16,4))
plt.plot(df_all.index, df_all["dBdt"], linewidth=0.6)
plt.axhline(0, color="gray", linewidth=0.8)
plt.ylabel("d|B|/dt (nT/s)")
plt.title("Aditya-L1 MAG d|B|/dt (Spacecraft frame)")
plt.grid(alpha=0.3)
plt.show()

df_all.to_pickle("./aditya-l1/data.pkl")