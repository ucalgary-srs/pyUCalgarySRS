import pyucalgarysrs
import pprint
import datetime

srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

start_dt = datetime.datetime(2023, 1, 1, 0, 0, 0)
end_dt = datetime.datetime(2023, 1, 1, 0, 59, 59)

# get urls
file_listing_obj = srs.data.get_urls("THEMIS_ASI_RAW", start_dt, end_dt, site_uid="atha")

# do fewer urls
file_listing_obj.urls = file_listing_obj.urls[0:2]

# download urls
print()
res = srs.data.download_using_urls(file_listing_obj, overwrite=True)

print()
print(res)
print()
pprint.pprint(res.__dict__)
print()
