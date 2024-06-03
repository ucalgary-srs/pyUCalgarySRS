import pyucalgarysrs

srs = pyucalgarysrs.PyUCalgarySRS()
srs.api_base_url = "https://api-staging.phys.ucalgary.ca"

sites = srs.data.list_sites("themis_asi")

print("\nFound %d sites\n" % (len(sites)))
for s in sites:
    print(s)
print()
