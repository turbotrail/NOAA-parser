import xarray as xr
ds = xr.open_dataset("./aditya-l1/L1_MAG91N18P1AL10000809024026019040426958_N00_0000_000761_V00.nc", decode_cf=True, mask_and_scale=True)

print(ds["Bx1_gse"].attrs)
print(ds["Bx1_gse"].values[:10])

