import pyucalgarysrs
import datetime
import pprint

# init
srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

# get dataset
print("\n[%s] Getting dataset ..." % (datetime.datetime.now()))
dataset = srs.data.get_dataset("TREX_SPECT_PROCESSED_V1")

# download data
print("\n[%s] Downloading data ..." % (datetime.datetime.now()))
start_dt = datetime.datetime(2018, 2, 8, 6, 0, 0)
end_dt = datetime.datetime(2018, 2, 8, 7, 59, 59)
site_uid = "luck"
download_obj = srs.data.download(dataset.name, start_dt, end_dt, site_uid=site_uid, progress_bar_disable=True)

# set list of files (we could do this using a glob too)
file_list = download_obj.filenames

# read data
print("\n[%s] Reading data ..." % (datetime.datetime.now()))
data = srs.data.readers.read_trex_spectrograph(file_list, n_parallel=1, dataset=dataset)
# data = srs.data.readers.read_trex_spectrograph(file_list, n_parallel=1)

print()
print(data)

print()
data.pretty_print()

print()
pprint.pprint(data.metadata[0])
