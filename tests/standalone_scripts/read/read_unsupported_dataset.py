import pyucalgarysrs
import datetime

# init
srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

# get dataset
print("\n[%s] Getting dataset ..." % (datetime.datetime.now()))
dataset = srs.data.get_dataset("THEMIS_ASI_RAW_ROW2")

# download data
print("\n[%s] Downloading data ..." % (datetime.datetime.now()))
start_dt = datetime.datetime(2023, 1, 1, 6, 0, 0)
end_dt = datetime.datetime(2023, 1, 1, 6, 4, 59)
site_uid = "atha"
download_obj = srs.data.download(dataset.name, start_dt, end_dt, site_uid=site_uid, progress_bar_disable=True)

# set list of files (we could do this using a glob too)
file_list = download_obj.filenames

# read data
print("\n[%s] Reading data ..." % (datetime.datetime.now()))
try:
    data = srs.data.read(dataset, file_list, n_parallel=1)
    print()
    print(data)
except pyucalgarysrs.SRSUnsupportedReadError as e:
    print("Expected error occurred\n\n[pyucalgarysrs.SRSUnsupportedReadError]: %s\n" % (str(e)))
