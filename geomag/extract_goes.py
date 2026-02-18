import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================
DATA_DIR = Path(".")  # folder containing .nc files
OUT_DIR = Path("csv_out")
OUT_DIR.mkdir(exist_ok=True)

# =========================
# PROCESS EACH NC FILE
# =========================
for nc_file in sorted(DATA_DIR.glob("*.nc")):
    print(f"\n📂 Processing {nc_file.name}")

    ds = xr.open_dataset(nc_file)

    # --- Extract OB GSM ---
    gsm = ds["b_ob_gsm"].values
    time = ds["time_ob"].values

    df = pd.DataFrame({
        "time": time,
        "Bx_GSM": gsm[:, 0],
        "By_GSM": gsm[:, 1],
        "Bz_GSM": gsm[:, 2],
        "B_total": ds["b_total"].values,
        "quality": ds["b_ob_quality"].values
    })

    print("RAW rows:", len(df))

    # --- Remove NaN-only regions ---
    df_valid = df.dropna(subset=["Bx_GSM", "By_GSM", "Bz_GSM"])
    print("VALID rows:", len(df_valid))

    # --- Gentle quality filtering ---
    df_clean = df_valid[
        df_valid["quality"].isna() | (df_valid["quality"] < 4)
    ]

    # --- Save per-day CLEAN 1-sec CSV ---
    base = nc_file.stem
    clean_csv = OUT_DIR / f"{base}_gsm_clean_1sec.csv"
    df_clean.to_csv(clean_csv, index=False)

    # =========================
    # 1‑MINUTE RMS DOWNSAMPLING
    # =========================
    df_rms = df_clean.copy()
    df_rms["time"] = pd.to_datetime(df_rms["time"], utc=True)
    df_rms = df_rms.set_index("time").sort_index()

    df_1min = df_rms.resample("1min").apply(
        lambda x: np.sqrt(np.nanmean(x**2))
    )

    df_1min = df_1min.dropna(subset=["Bx_GSM", "By_GSM", "Bz_GSM"])

    # Recompute |B| from RMS vectors
    df_1min["B_total_calc"] = np.sqrt(
        df_1min["Bx_GSM"]**2 +
        df_1min["By_GSM"]**2 +
        df_1min["Bz_GSM"]**2
    )

    rms_csv = OUT_DIR / f"{base}_gsm_1min_rms.csv"
    df_1min.reset_index().to_csv(rms_csv, index=False)

    print(f"✅ Wrote {clean_csv.name}")
    print(f"✅ Wrote {rms_csv.name}")

print("\n🎉 All files processed.")