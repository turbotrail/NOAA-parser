import pandas as pd
import matplotlib.pyplot as plt


OBSERVATORY_ID="GUA"
CSV_FILE =f"data{OBSERVATORY_ID}_2025-11-09T00:00:00Z_2026-02-16T00:00:00Z.csv" #"data.csv"



# =========================
# LOAD DATA
# =========================
df = pd.read_csv(
    CSV_FILE,
    usecols=["DATE", "TIME", OBSERVATORY_ID+"X", OBSERVATORY_ID+"Y", OBSERVATORY_ID+"Z", OBSERVATORY_ID+"F"],
    dtype={
        OBSERVATORY_ID+"X": "float32",
        OBSERVATORY_ID+"Y": "float32",
        OBSERVATORY_ID+"Z": "float32",
        OBSERVATORY_ID+"F": "float32"
    }
)

import numpy as np

# Replace sentinel values with NaN
SENTINEL = 99999.0
cols = [OBSERVATORY_ID+"X", OBSERVATORY_ID+"Y", OBSERVATORY_ID+"Z", OBSERVATORY_ID+"F"]

df[cols] = df[cols].replace(SENTINEL, np.nan)

# Combine DATE + TIME → timestamp
df["timestamp"] = pd.to_datetime(df["DATE"] + " " + df["TIME"], utc=True)

df.set_index("timestamp", inplace=True)

print(f"Loaded {len(df):,} rows")

# =========================
# PLOTTING
# =========================
plt.figure(figsize=(16, 8))

plt.plot(df.index, df[OBSERVATORY_ID+"X"], label=OBSERVATORY_ID+"X", linewidth=0.6)
plt.plot(df.index, df[OBSERVATORY_ID+"Y"], label=OBSERVATORY_ID+"Y", linewidth=0.6)
plt.plot(df.index, df[OBSERVATORY_ID+"Z"], label=OBSERVATORY_ID+"Z", linewidth=0.6)

plt.title(OBSERVATORY_ID+" Geomagnetic Field Components")
plt.xlabel("Time (UTC)")
plt.ylabel("Magnetic Field (nT)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 8))

plt.plot(df.index, df[OBSERVATORY_ID+"F"], label=OBSERVATORY_ID+"F", linewidth=0.6)
plt.title(OBSERVATORY_ID+" Total Geomagnetic Field")
plt.xlabel("Time (UTC)")
plt.ylabel("Magnetic Field (nT)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


dXdt = df[OBSERVATORY_ID+"X"].diff() / 60  # nT/s for 1-min data

plt.figure(figsize=(16,4))
plt.plot(df.index, dXdt, linewidth=0.6)
plt.axhline(0, color="black", alpha=0.3)
plt.title(OBSERVATORY_ID+" dX/dt (GIC-relevant signal)")
plt.ylabel("nT/s")
plt.show()