import datetime
import pyucalgarysrs

srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

output = pyucalgarysrs.ATMInverseOutputFlags()
output.set_all_true()

timestamp = datetime.datetime(2022, 1, 1, 6, 0, 0)
lat = 51.04
lon = -100.0
intensity_4278 = 2302.6
intensity_5577 = 11339.5
intensity_6300 = 528.3
intensity_8446 = 427.4

result = srs.models.atm.inverse(timestamp, lat, lon, intensity_4278, intensity_5577, intensity_6300, intensity_8446, output)

print()
print(result)
print()
result.pretty_print()
print()
