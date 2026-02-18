import xarray as xr
from pathlib import Path

# Path to ONE NetCDF file
nc_file = Path("goes19-mps/ops_seis-l1b-mpsl_g19_d20260118_v0-0-0.nc")

ds = xr.open_dataset(nc_file)

print("\n=== DATA VARIABLES ===")
for name, var in ds.data_vars.items():
    print(f"{name:40s} {tuple(var.dims)} {var.dtype}")

print("\n=== COORDS ===")
for name, coord in ds.coords.items():
    print(f"{name:40s} {tuple(coord.dims)} {coord.dtype}")

print("\n=== DIMENSIONS ===")
for dim, size in ds.dims.items():
    print(f"{dim:30s} {size}")

print("\n=== GLOBAL ATTRIBUTES ===")
for k, v in ds.attrs.items():
    print(f"{k}: {v}")