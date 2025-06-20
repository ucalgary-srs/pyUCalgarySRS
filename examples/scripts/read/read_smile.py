import pyucalgarysrs
import datetime
import pprint

# init
srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

# get dataset
print("\n[%s] Getting dataset ..." % (datetime.datetime.now()))
dataset = srs.data.get_dataset("SMILE_ASI_RAW")

# download data
print("\n[%s] Downloading data ..." % (datetime.datetime.now()))
start_dt = datetime.datetime(2025, 3, 1, 6, 0, 0)
end_dt = datetime.datetime(2025, 3, 1, 6, 4, 59)
site_uid = "gill"
download_obj = srs.data.download(dataset.name, start_dt, end_dt, site_uid=site_uid, progress_bar_disable=True)

# set list of files (we could do this using a glob too)
file_list = download_obj.filenames

# read data
print("\n[%s] Reading data ..." % (datetime.datetime.now()))
data = srs.data.read(dataset, file_list, n_parallel=1)
print()
data.pretty_print()

# read data with multiprocessing
print("\n[%s] Reading data with n_parallel ..." % (datetime.datetime.now()))
data = srs.data.read(dataset, file_list, n_parallel=2)
print()
data.pretty_print()

# read data with no metadata
print("\n[%s] Reading data with no_metadata ..." % (datetime.datetime.now()))
data = srs.data.read(dataset, file_list, no_metadata=True)
print()
data.pretty_print()

# read data with first record
print("\n[%s] Reading data with first_record ..." % (datetime.datetime.now()))
data = srs.data.read(dataset, file_list, first_record=True)
print()
data.pretty_print()

# read data with start time and end time
start_dt = datetime.datetime(2025, 3, 1, 6, 2)
end_dt = datetime.datetime(2025, 3, 1, 6, 3)
print("\n[%s] Reading data with start+end times ..." % (datetime.datetime.now()))
data = srs.data.read(dataset, file_list, start_time=start_dt, end_time=end_dt)
print()
data.pretty_print()
print()
pprint.pprint(data.timestamp[0:5])
pprint.pprint(data.timestamp[-5:])

print("\n[%s] Reading data with start time ..." % (datetime.datetime.now()))
data = srs.data.read(dataset, file_list, start_time=start_dt)
print()
data.pretty_print()
print()
pprint.pprint(data.timestamp[0:5])
pprint.pprint(data.timestamp[-5:])

print("\n[%s] Reading data with end time ..." % (datetime.datetime.now()))
data = srs.data.read(dataset, file_list, end_time=end_dt)
print()
data.pretty_print()
print()
pprint.pprint(data.timestamp[0:5])
pprint.pprint(data.timestamp[-5:])
