import pyucalgarysrs
import datetime
import os

# init
srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

# get dataset
print("\n[%s] Getting dataset ..." % (datetime.datetime.now()))
dataset = srs.data.list_datasets("THEMIS_ASI_SKYMAP_IDLSAV")[0]

# download data
print("\n[%s] Downloading data ..." % (datetime.datetime.now()))
start_dt = datetime.datetime(2000, 1, 1, 0, 0, 0)
end_dt = datetime.datetime.now()
download_obj = srs.data.download(dataset.name, start_dt, end_dt)

# set list of files (we could do this using a glob too)
file_list = download_obj.filenames

# read data
print("\n[%s] Reading data ..." % (datetime.datetime.now()))
for f in file_list:
    print("[%s]  Reading %s" % (datetime.datetime.now(), os.path.basename(f)))
    data = srs.data.readers.read_skymap(f, dataset=dataset)
