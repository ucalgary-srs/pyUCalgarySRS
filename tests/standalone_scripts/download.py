import pyucalgarysrs
import datetime

srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

start_dt = datetime.datetime(2023, 1, 1, 0, 0, 0)
end_dt = datetime.datetime(2023, 1, 1, 0, 59, 59)

print()
res = srs.data.download("REGO_CALIBRATION_FLATFIELD_IDLSAV", start_dt, end_dt, site_uid="atha", overwrite=True)

print()
print(res)
print()
res.pretty_print()
print()
