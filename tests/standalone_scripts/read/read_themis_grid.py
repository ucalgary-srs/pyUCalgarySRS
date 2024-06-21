import pyucalgarysrs
import datetime

# init
srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

# get dataset
print("\n[%s] Getting dataset ..." % (datetime.datetime.now()))
dataset = srs.data.list_datasets("THEMIS_ASI_GRID_MOSV001")[0]

# download data
print("\n[%s] Downloading data ..." % (datetime.datetime.now()))
start_dt = datetime.datetime(2022, 2, 3, 6, 0, 0)
end_dt = datetime.datetime(2022, 2, 3, 6, 4, 59)
download_obj = srs.data.download(dataset.name, start_dt, end_dt, progress_bar_disable=True)

# set list of files (we could do this using a glob too)
file_list = download_obj.filenames

# read data
print("\n[%s] Reading data ..." % (datetime.datetime.now()))
data = srs.data.read(dataset, file_list, n_parallel=2)

print()
print(data)
