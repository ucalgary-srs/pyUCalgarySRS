import pyucalgarysrs
import datetime

# init
srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

# get dataset
print("\n[%s] Getting dataset ..." % (datetime.datetime.now()))
dataset = srs.data.get_dataset("TREX_BLUE_CALIBRATION_FLATFIELD_IDLSAV")

# download data
print("\n[%s] Downloading data ..." % (datetime.datetime.now()))
start_dt = datetime.datetime(2017, 1, 1, 0, 0, 0)
end_dt = datetime.datetime(2022, 12, 31, 23, 59, 59)
download_obj = srs.data.download(dataset.name, start_dt, end_dt)

# set list of files (we could do this using a glob too)
file_list = download_obj.filenames

# read data
print("\n[%s] Reading data ..." % (datetime.datetime.now()))
data = srs.data.read(dataset, file_list, n_parallel=2)

print()
print(data.data[0])
print()
