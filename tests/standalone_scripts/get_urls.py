import pyucalgarysrs
import pprint
import datetime

srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

start_dt = datetime.datetime(2023, 1, 1, 0, 0, 0)
end_dt = datetime.datetime(2023, 1, 1, 0, 59, 59)

res = srs.data.get_urls("THEMIS_ASI_RAW", start_dt, end_dt, site_uid="atha")

print()
print(res)
print()
pprint.pprint(res.__dict__)
print()
