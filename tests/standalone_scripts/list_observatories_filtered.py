import pyucalgarysrs

srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

observatories = srs.data.list_observatories("themis_asi", uid="fs")

print("\nFound %d observatories matching the uid filter\n" % (len(observatories)))
for o in observatories:
    print(o)
print()
