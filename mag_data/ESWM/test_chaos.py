# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chaosmagpy"
# ]
# ///
import chaosmagpy as cp
import numpy as np

shc_file = "SW_OPER_MLI_SHA_2C_00000000T000000_99999999T999999_1101/SW_OPER_MLI_SHA_2C_00000000T000000_99999999T999999_1101.shc"

try:
    model = cp.load_shcfile(shc_file)
    print("Model loaded.")
    print("Methods:", dir(model))
except Exception as e:
    print("Error:", e)
