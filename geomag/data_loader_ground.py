# curl -X 'GET' \
#   'https://geomag.usgs.gov/ws/data/?id=BOU&elements=X&elements=Y&elements=Z&elements=F&sampling_period=60&type=adjusted&format=iaga2002&data_host=edgecwb.usgs.gov' \
#   -H 'accept: application/json'

# curl -X 'GET' \
#   'https://geomag.usgs.gov/ws/data/?id=BOU&starttime=2026-02-17T00%3A00%3A00Z&endtime=2026-02-17T06%3A00%3A00Z&elements=X&elements=Y&elements=Z&elements=F&sampling_period=60&type=adjusted&format=iaga2002&data_host=edgecwb.usgs.gov' \
#   -H 'accept: application/json'


import requests
import json

BASE_URL = "https://geomag.usgs.gov/ws/data/"

# =========================
# CONFIG
# =========================
OBSERVATORY_ID = "CMO"

START_TIME = "2026-01-18T00:00:00Z"
END_TIME   = "2026-01-23T00:00:00Z"

ELEMENTS = ["X", "Y", "Z", "F"]   # magnetic field components
SAMPLING_PERIOD = 60             # seconds
DATA_TYPE = "adjusted"            # adjusted | preliminary | definitive
FORMAT = "iaga2002"               # iaga2002 | json | csv
DATA_HOST = "edgecwb.usgs.gov"

# =========================
# REQUEST PARAMS
# =========================
params = {
    "id": OBSERVATORY_ID,
    "starttime": START_TIME,
    "endtime": END_TIME,
    "elements": ELEMENTS,          # requests handles repeated params
    "sampling_period": SAMPLING_PERIOD,
    "type": DATA_TYPE,
    "format": FORMAT,
    "data_host": DATA_HOST,
}

# =========================
# API CALL
# =========================
response = requests.get(BASE_URL, params=params, timeout=30)

# =========================
# HANDLE RESPONSE
# =========================
response.raise_for_status()

data = response.text 
if FORMAT == "json":
    data = json.loads(data) # iaga2002 is text-based
    with open(f"./ground/data{OBSERVATORY_ID}_{START_TIME}_{END_TIME}.txt", "w") as f:
        f.write(data)
else:
    with open(f"./ground/data{OBSERVATORY_ID}_{START_TIME}_{END_TIME}.txt", "w") as f:
        f.write(data)




