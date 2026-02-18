

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
CSV_DIR = Path("csv_out")
PATTERN = "*_gsm_1min_rms.csv"

# =========================
# LOAD + STITCH CSV FILES
# =========================
dfs = []

for csv_file in sorted(CSV_DIR.glob(PATTERN)):
    print(f"📄 Loading {csv_file.name}")
    df = pd.read_csv(csv_file, parse_dates=["time"])
    dfs.append(df)

if not dfs:
    raise RuntimeError("No 1-min RMS CSV files found.")

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.sort_values("time")
df_all = df_all.set_index("time")

print(f"\n✅ Combined rows: {len(df_all)}")

# =========================
# DERIVED QUANTITIES
# =========================
# |B| magnitude from RMS vectors (already computed, but keep safe)
df_all["B_mag"] = np.sqrt(
    df_all["Bx_GSM"]**2 +
    df_all["By_GSM"]**2 +
    df_all["Bz_GSM"]**2
)

# dB/dt (nT/min)
df_all["dBx_dt"] = df_all["Bx_GSM"].diff()
df_all["dBy_dt"] = df_all["By_GSM"].diff()
df_all["dBz_dt"] = df_all["Bz_GSM"].diff()

df_all["dBdt"] = np.sqrt(
    df_all["dBx_dt"]**2 +
    df_all["dBy_dt"]**2 +
    df_all["dBz_dt"]**2
)

# =========================
# PLOTTING
# =========================
plt.figure(figsize=(14, 10))

# --- Vector components ---
ax1 = plt.subplot(3, 1, 1)
df_all[["Bx_GSM", "By_GSM", "Bz_GSM"]].plot(ax=ax1)
ax1.set_title("GOES-19 GSM Magnetic Field (1-min RMS)")
ax1.set_ylabel("nT")
ax1.grid(True)

# --- |B| magnitude ---
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
df_all["B_mag"].plot(ax=ax2)
ax2.set_title("|B| Total Field")
ax2.set_ylabel("nT")
ax2.grid(True)

# --- dB/dt ---
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
df_all["dBdt"].plot(ax=ax3)
ax3.set_title("dB/dt (1-min, storm activity proxy)")
ax3.set_ylabel("nT / min")
ax3.set_xlabel("UTC Time")
ax3.grid(True)

plt.tight_layout()
plt.show()

# =========================
# OPTIONAL: SAVE STITCHED CSV
# =========================
df_all.reset_index().to_csv(
    CSV_DIR / "goes19_gsm_1min_rms_stitched.csv",
    index=False
)

print("📊 Plot complete.")