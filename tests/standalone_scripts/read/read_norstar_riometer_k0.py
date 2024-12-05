import pprint
import datetime
import pyucalgarysrs

# init
srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

# get dataset
print("\n[%s] Getting dataset ..." % (datetime.datetime.now()))
dataset = srs.data.get_dataset("NORSTAR_RIOMETER_K0_TXT")

# get the data
print("\n[%s] Downloading data ..." % (datetime.datetime.now()))
start_dt = datetime.datetime(2009, 1, 1, 0, 0, 0)
end_dt = datetime.datetime(2009, 1, 5, 23, 59, 59)
site_uid = "gill"
download_obj = srs.data.download(dataset.name, start_dt, end_dt, site_uid=site_uid, progress_bar_disable=True)

# set list of files (we could do this using a glob too)
file_list = download_obj.filenames

# read data
print("\n[%s] Reading data ..." % (datetime.datetime.now()))
data = srs.data.read(dataset, file_list, n_parallel=1)
print()
data.pretty_print()
print()
data.data[0].pretty_print()

# read data with multiprocessing
print("\n[%s] Reading data with n_parallel ..." % (datetime.datetime.now()))
data = srs.data.read(dataset, file_list, n_parallel=2)
print()
data.pretty_print()
print()
data.data[0].pretty_print()

# read data with no metadata
print("\n[%s] Reading data with no_metadata ..." % (datetime.datetime.now()))
data = srs.data.read(dataset, file_list, no_metadata=True)
print()
data.pretty_print()
print()
data.data[0].pretty_print()

# read data with start time and end time
start_dt = datetime.datetime(2009, 1, 1, 12, 0, 0)
end_dt = datetime.datetime(2009, 1, 1, 13, 29, 59)
print("\n[%s] Reading data with start+end times ..." % (datetime.datetime.now()))
data = srs.data.read(dataset, file_list, start_time=start_dt, end_time=end_dt)
print()
data.pretty_print()
print()
pprint.pprint(data.data[0].timestamp[0:5])
pprint.pprint(data.data[0].timestamp[-5:])
pprint.pprint(data.data[-1].timestamp[0:5])
pprint.pprint(data.data[-1].timestamp[-5:])

print("\n[%s] Reading data with start time ..." % (datetime.datetime.now()))
data = srs.data.read(dataset, file_list, start_time=start_dt)
print()
data.pretty_print()
print()
pprint.pprint(data.data[0].timestamp[0:5])
pprint.pprint(data.data[0].timestamp[-5:])
pprint.pprint(data.data[-1].timestamp[0:5])
pprint.pprint(data.data[-1].timestamp[-5:])

print("\n[%s] Reading data with end time ..." % (datetime.datetime.now()))
data = srs.data.read(dataset, file_list, end_time=end_dt)
print()
data.pretty_print()
print()
pprint.pprint(data.data[0].timestamp[0:5])
pprint.pprint(data.data[0].timestamp[-5:])
pprint.pprint(data.data[-1].timestamp[0:5])
pprint.pprint(data.data[-1].timestamp[-5:])
