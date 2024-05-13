# type: ignore
import pyucalgarysrs
import datetime
import os
from typing import List

# init
srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

# get dataset
print("\n[%s] Getting dataset ..." % (datetime.datetime.now()))
dataset = srs.data.list_datasets("TREX_NIR_CALIBRATION_RAYLEIGHS_IDLSAV")[0]

# download data
print("\n[%s] Downloading data ..." % (datetime.datetime.now()))
start_dt = datetime.datetime(2017, 1, 1, 0, 0, 0)
end_dt = datetime.datetime.now()
download_obj = srs.data.download(dataset.name, start_dt, end_dt)

# set list of files (we could do this using a glob too)
file_list = download_obj.filenames

# read data
print("\n[%s] Reading data ..." % (datetime.datetime.now()))
data: List[pyucalgarysrs.Calibration] = srs.data.read(dataset, file_list, n_parallel=2)

print("\n[%s] Have %d calibration objects\n" % (datetime.datetime.now(), len(data)))
print(data[0])
print()

for d in data:
    print("%-50s%s" % (os.path.basename(d.filename), d.rayleighs_perdn_persecond))
