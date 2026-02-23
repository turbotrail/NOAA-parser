import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
ADITYA_PKL = Path("./aditya-l1/data.pkl")
GOES_PKL   = Path("./goes_mag/data.pkl")
CMO_PKL    = Path("./ground/data.pkl")

COLUMN_MAP = {
    "aditya": "dBdt",
    "goes":   "dBdt",
    "cmo":    None,  # auto-detect
}

MAX_LAG_MIN = 180   # minutes for correlation search
ONSET_SIGMA = 2.5   # threshold in std devs

# =========================
# LOAD DATA
# =========================
aditya = pd.read_pickle(ADITYA_PKL)
goes   = pd.read_pickle(GOES_PKL)
cmo    = pd.read_pickle(CMO_PKL)

# Auto-detect or compute CMO dX/dt
possible_cmo_cols = [c for c in cmo.columns if "dx" in c.lower() and "dt" in c.lower()]
if possible_cmo_cols:
    COLUMN_MAP["cmo"] = possible_cmo_cols[0]
    print(f"Using existing CMO column: {COLUMN_MAP['cmo']}")
else:
    # Fallback: compute dX/dt from X-component
    x_cols = [c for c in cmo.columns if c.lower().endswith("x")]
    if not x_cols:
        raise RuntimeError(
            f"No X-component column found to compute dX/dt. Available columns: {list(cmo.columns)}"
        )
    x_col = x_cols[0]
    print(f"Computing dX/dt from column: {x_col}")
    # Ensure datetime index before diff
    cmo = cmo.copy()
    # Assume 1-minute cadence; convert to nT/s
    cmo["dXdt"] = cmo[x_col].diff() / 60.0
    COLUMN_MAP["cmo"] = "dXdt"

# Ensure UTC index
for df in (aditya, goes, cmo):
    df.index = pd.to_datetime(df.index, utc=True)

# =========================
# ALIGN TIME WINDOW
# =========================
start = max(aditya.index.min(), goes.index.min(), cmo.index.min())
end   = min(aditya.index.max(), goes.index.max(), cmo.index.max())

aditya = aditya.loc[start:end]
goes   = goes.loc[start:end]
cmo    = cmo.loc[start:end]

# =========================
# NORMALIZATION
# =========================
def normalize(series):
    return (series - series.mean()) / series.std()

A = normalize(aditya[COLUMN_MAP["aditya"]])
G = normalize(goes[COLUMN_MAP["goes"]])
C = normalize(cmo[COLUMN_MAP["cmo"]])

# =========================
# ONSET DETECTION
# =========================
def detect_onset_sustained(series, sigma=2.5, minutes=5):
    """
    Detect onset as sustained activity above threshold for a given duration.
    This avoids triggering on isolated noise spikes.
    """
    threshold = series.std() * sigma
    mask = series.abs() > threshold
    sustained = mask.rolling(minutes, center=False).sum() >= minutes
    return sustained[sustained].index[0]

def detect_main_onset(series, percentile=95):
    """
    Detect main event onset based on high-percentile activity.
    Represents the main coupling / shock phase.
    """
    thresh = np.percentile(series.abs().dropna(), percentile)
    return series[series.abs() > thresh].index[0]

onset_aditya = detect_onset_sustained(A, ONSET_SIGMA)
onset_goes   = detect_onset_sustained(G, ONSET_SIGMA)
onset_cmo    = detect_onset_sustained(C, ONSET_SIGMA)

main_onset_aditya = detect_main_onset(A)
main_onset_goes   = detect_main_onset(G)
main_onset_cmo    = detect_main_onset(C)

# =========================
# LAG COMPUTATION
# =========================
def compute_lag(x, y, max_lag):
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = [
        x.shift(lag).corr(y)
        for lag in lags
    ]
    best_lag = lags[np.nanargmax(corrs)]
    return best_lag, np.nanmax(corrs)

lag_A_G, corr_A_G = compute_lag(A, G, MAX_LAG_MIN)
lag_G_C, corr_G_C = compute_lag(G, C, MAX_LAG_MIN)
lag_A_C, corr_A_C = compute_lag(A, C, MAX_LAG_MIN)

# =========================
# RESULTS
# =========================
print("\n===== ONSET TIMES (UTC) =====")
print(f"Aditya-L1 onset : {onset_aditya}")
print(f"GOES-19 onset   : {onset_goes}")
print(f"CMO onset       : {onset_cmo}")

print("\n===== ONSET LAGS =====")
print(f"Aditya → GOES : {(onset_goes - onset_aditya).total_seconds()/60:.1f} min")
print(f"GOES → CMO    : {(onset_cmo - onset_goes).total_seconds()/60:.1f} min")
print(f"Aditya → CMO  : {(onset_cmo - onset_aditya).total_seconds()/60:.1f} min")

print("\n===== CROSS-CORRELATION LAGS =====")
print(f"Aditya → GOES : {lag_A_G:+d} min (corr={corr_A_G:.2f})")
print(f"GOES → CMO    : {lag_G_C:+d} min (corr={corr_G_C:.2f})")
print(f"Aditya → CMO  : {lag_A_C:+d} min (corr={corr_A_C:.2f})")

# =========================
# TIMELINE PLOT (ZOOMED + MULTI-PANEL)
# =========================
center = main_onset_goes
window = pd.Timedelta(hours=6)

x_start = center - window
x_end   = center + window

A_zoom = A.loc[center-window : center+window]
G_zoom = G.loc[center-window : center+window]
C_zoom = C.loc[center-window : center+window]

fig, axes = plt.subplots(
    nrows=4, ncols=1,
    figsize=(16,10),
    sharex=True,
    gridspec_kw={"height_ratios": [2,1,1,1]}
)

# ---- Combined plot ----
ax = axes[0]
ax.plot(A_zoom.index, A_zoom, label="Aditya-L1 d|B|/dt")
ax.plot(G_zoom.index, G_zoom, label="GOES-19 d|B|/dt")
ax.plot(C_zoom.index, C_zoom, label="CMO dX/dt")

# Sustained onsets (dashed)
ax.axvline(onset_aditya, color="tab:blue", linestyle="--", alpha=0.5, label="Aditya sustained")
ax.axvline(onset_goes,   color="k",        linestyle="--", alpha=0.7, label="GOES sustained")
ax.axvline(onset_cmo,    color="tab:green",linestyle="--", alpha=0.5, label="CMO sustained")

# Main onsets (solid)
ax.axvline(main_onset_aditya, color="tab:blue", linestyle="-", linewidth=2, label="Aditya main")
ax.axvline(main_onset_goes,   color="k",        linestyle="-", linewidth=2, label="GOES main")
ax.axvline(main_onset_cmo,    color="tab:green",linestyle="-", linewidth=2, label="CMO main")

ax.set_title("Upstream → Magnetosphere → Ground dB/dt Timeline")
ax.set_ylabel("Normalized units")
ax.grid(alpha=0.3)
ax.legend(
    handles=[
        plt.Line2D([], [], color="tab:blue", lw=2, label="Aditya-L1 d|B|/dt"),
        plt.Line2D([], [], color="tab:orange", lw=2, label="GOES-19 d|B|/dt"),
        plt.Line2D([], [], color="tab:green", lw=2, label="CMO dX/dt"),
        plt.Line2D([], [], color="k", lw=2, label="Main onset"),
    ],
    fontsize=9,
    ncol=4,
)

ax.set_xlim(x_start, x_end)


# Light smoothing for individual panels (presentation only)
A_plot = A_zoom.rolling(3, center=True).mean()
G_plot = G_zoom.rolling(3, center=True).mean()
C_plot = C_zoom.rolling(3, center=True).mean()

# ---- Individual panels ----
axes[1].plot(A_plot.index, A_plot, color="tab:blue")
axes[1].axvline(main_onset_aditya, color="tab:blue", linewidth=2)
axes[1].set_ylabel("Aditya L1")
axes[1].grid(alpha=0.3)
axes[1].set_xlim(x_start, x_end)

axes[2].plot(G_plot.index, G_plot, color="tab:orange")
axes[2].axvline(main_onset_goes, color="k", linewidth=2)
axes[2].set_ylabel("GOES 19")
axes[2].grid(alpha=0.3)
axes[2].set_xlim(x_start, x_end)

axes[3].plot(C_plot.index, C_plot, color="tab:green")
axes[3].axvline(main_onset_cmo, color="tab:green", linewidth=2)
axes[3].set_ylabel("CMO (Alaska Ground station)")
axes[3].set_xlabel("UTC")
axes[3].grid(alpha=0.3)
axes[3].set_xlim(x_start, x_end)

plt.tight_layout()
plt.show()