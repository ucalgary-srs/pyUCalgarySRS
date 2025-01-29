import pyucalgarysrs
import datetime

srs = pyucalgarysrs.PyUCalgarySRS()

dataset_name = "THEMIS_ASI_RAW"
start_dt = datetime.datetime(2023, 1, 1, 6, 0, 0)
end_dt = datetime.datetime(2023, 1, 1, 6, 59, 59)
site_uid = "atha"
res = srs.data.download(dataset_name, start_dt, end_dt, site_uid=site_uid, overwrite=True)

print()
print(res)
print()
res.pretty_print()
print()
