import pyucalgarysrs
import datetime
import pprint

# init
srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

# get dataset
print("\n[%s] Getting dataset ..." % (datetime.datetime.now()))
dataset = srs.data.list_datasets("TREX_NIR_SKYMAP_IDLSAV")[0]

# download data
print("\n[%s] Downloading data ..." % (datetime.datetime.now()))
start_dt = datetime.datetime(2022, 1, 1, 0, 0, 0)
end_dt = datetime.datetime(2022, 12, 31, 23, 59, 59)
site_uid = "luck"
download_obj = srs.data.download(dataset.name, start_dt, end_dt, site_uid=site_uid, progress_bar_disable=True)

# set list of files (we could do this using a glob too)
file_list = download_obj.filenames

# read data
print("\n[%s] Reading data ..." % (datetime.datetime.now()))
data = srs.data.readers.read_skymap(file_list, n_parallel=2, dataset=dataset)

print()
if (data is not None):
    pprint.pprint(data[0].__dict__)

print()
print(data)
