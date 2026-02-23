from astropy.io import fits
import numpy as np
file_name="./image/SUT_T26_0392_001842_Lev1.0_2026-02-19T08.00.23.691_08B1NB04.fits"
hdul = fits.open(file_name)

data = hdul[0].data
header = hdul[0].header

print(data.shape)
print(data.dtype)
data_disp = np.flipud(data)
# data_disp = np.max(data_disp) - data_disp

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 8))
plt.imshow(
    data_disp,
    cmap="inferno",
    vmin=np.percentile(data_disp, 5),
    vmax=np.percentile(data_disp, 95)
)
plt.axis("off")
plt.title("SUIT NB04 – Website Style (False Color)")
plt.show()


hdul = fits.open(file_name)
hdr = hdul[0].header

print(hdr["WAVELNTH"])   # or WAVELENG
