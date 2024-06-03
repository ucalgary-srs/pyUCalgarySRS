import pyucalgarysrs

srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

sites = srs.data.list_sites("themis_asi", uid="fs")

print("\nFound %d datasets matching the uid filter\n" % (len(sites)))
for s in sites:
    print(s)
print()
