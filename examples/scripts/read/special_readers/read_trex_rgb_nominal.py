import pyucalgarysrs
import datetime

# init
srs = pyucalgarysrs.PyUCalgarySRS()

# get dataset
print("\n[%s] Getting dataset ..." % (datetime.datetime.now()))
dataset = srs.data.get_dataset("TREX_RGB_RAW_NOMINAL")

# download data
print("\n[%s] Downloading data ..." % (datetime.datetime.now()))
start_dt = datetime.datetime(2023, 1, 1, 6, 0, 0)
end_dt = datetime.datetime(2023, 1, 1, 6, 4, 59)
site_uid = "gill"
download_obj = srs.data.download(dataset.name, start_dt, end_dt, site_uid=site_uid, progress_bar_disable=True)

# set list of files (we could do this using a glob too)
file_list = download_obj.filenames

# read data
print("\n[%s] Reading data ..." % (datetime.datetime.now()))
data = srs.data.readers.read_trex_rgb(file_list, n_parallel=2, dataset=dataset)

print()
print(data)
